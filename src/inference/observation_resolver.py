"""Pearls AQI Predictor - Unified Feature Observation Resolver.

Resolves feature observations with explicit hierarchy:
1. Hopsworks Cloud Feature Store (when FEATURE_SOURCE_MODE in {'auto', 'hopsworks'})
2. Local committed bootstrap feature vector (fallback in 'auto' or strictly in 'bootstrap')

Guarantees:
- Single observation contract across /current, /forecast, /explain
- Strict schema validation (114 canonical features in exact order)
- Dynamic freshness calculation on every request
- Bounded short-TTL caching (60s) with thread safety
- Dynamic derivation of dominant pollutant (never defaulted)
- Zero module-level Hopsworks imports (lazy on demand)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import os
import threading
import time
from typing import Any
import numpy as np
import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_FEATURE_GROUP_NAME,
    HOPSWORKS_FEATURE_GROUP_VERSION,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT,
)
from src.exceptions import FeatureStoreError, ValidationError
from src.feature_pipeline.aqi_calculator import calculate_overall_aqi
from src.inference.runtime_resolver import RuntimeAssetResolver
from src.logger import logger


@dataclass
class ResolvedFeatureObservation:
    """Immutable single source of truth for an inference/telemetry observation."""
    features: pd.DataFrame          # (1, 114) DataFrame with canonical columns in exact schema order
    features_array: np.ndarray      # (1, 114) float32 numpy array
    observed_at: datetime           # Physical UTC datetime of observation
    dt: int                         # Epoch integer
    current_aqi: float              # Physical EPA AQI at time T
    source: str                     # "hopsworks" | "bootstrap"
    cloud_active: bool              # True if from Hopsworks
    fallback_active: bool = False   # True if fallback was triggered
    fallback_reason: str | None = None
    dominant_pollutant: str | None = None
    pollutants: dict[str, float] = field(default_factory=dict)
    weather: dict[str, float] = field(default_factory=dict)

    def get_age_hours(self, now: datetime | None = None) -> float:
        """Calculate observation age in hours relative to UTC now."""
        now_utc = now or datetime.now(timezone.utc)
        obs = self.observed_at if self.observed_at.tzinfo else self.observed_at.replace(tzinfo=timezone.utc)
        return round(max(0.0, (now_utc - obs).total_seconds() / 3600.0), 2)

    def is_stale(self, now: datetime | None = None) -> bool:
        """Dynamically evaluate whether observation age exceeds the 3-hour threshold."""
        return self.get_age_hours(now) > 3.0


class FeatureObservationResolver:
    """Manages observation retrieval with caching and fallback hierarchy."""

    def __init__(
        self,
        mode: str | None = None,
        cache_ttl_seconds: float = 60.0,
        runtime_resolver: RuntimeAssetResolver | None = None,
        location_id: str = "lahore",
    ) -> None:
        """Initialize resolver.

        Args:
            mode: Feature source mode ('auto', 'hopsworks', or 'bootstrap').
            cache_ttl_seconds: In-memory observation cache TTL in seconds (default 60s).
            runtime_resolver: Optional RuntimeAssetResolver instance.
            location_id: Entity location identifier for online queries (default 'lahore').
        """
        self.mode = (mode or os.environ.get("FEATURE_SOURCE_MODE", "auto")).strip().lower()
        if self.mode not in {"auto", "hopsworks", "bootstrap"}:
            raise ValidationError(
                f"Invalid FEATURE_SOURCE_MODE '{self.mode}': must be 'auto', 'hopsworks', or 'bootstrap'"
            )
        self.cache_ttl_seconds = cache_ttl_seconds
        self.runtime_resolver = runtime_resolver or RuntimeAssetResolver()
        self.location_id = location_id
        self._cached_observation: ResolvedFeatureObservation | None = None
        self._cache_timestamp: float = 0.0
        self._lock = threading.Lock()
        self._last_source: str = "uninitialized"
        self._last_error: str | None = None

    def _fetch_hopsworks(self) -> ResolvedFeatureObservation:
        """Fetch and validate the latest online vector specifically for location_id from Hopsworks."""
        api_key = os.getenv("HOPSWORKS_API_KEY", HOPSWORKS_API_KEY)
        if not api_key or api_key.strip() == "":
            raise FeatureStoreError("Hopsworks API key is not configured in environment.")

        project_name = os.getenv("HOPSWORKS_PROJECT", HOPSWORKS_PROJECT)
        host = os.getenv("HOPSWORKS_HOST", HOPSWORKS_HOST)

        # Lazy import of hopsworks SDK to avoid import-time overhead
        try:
            import hopsworks
        except ImportError as e:
            raise FeatureStoreError(f"Hopsworks SDK is not installed in the environment: {e}") from e

        try:
            project = hopsworks.login(
                api_key_value=api_key.strip(),
                project=project_name,
                host=host,
            )
            fs = project.get_feature_store()
            fg = fs.get_feature_group(
                name=HOPSWORKS_FEATURE_GROUP_NAME,
                version=HOPSWORKS_FEATURE_GROUP_VERSION,
            )

            # Query online store specifically for location_id="lahore"
            cloud_df = fg.select_all().filter(fg.location_id == self.location_id).read(online=True)
            if cloud_df is None or cloud_df.empty:
                raise FeatureStoreError(
                    f"Hopsworks online query returned empty dataset for location_id='{self.location_id}'"
                )
        except FeatureStoreError:
            raise
        except Exception as e:
            raise FeatureStoreError(f"Hopsworks online connection or query failed: {e}") from e

        # Sort by event_time dt descending to ensure latest observation is used
        if "dt" in cloud_df.columns:
            cloud_df = cloud_df.sort_values(by="dt", ascending=False)
        latest_row = cloud_df.iloc[[0]].copy()

        # Timestamp validation
        if "dt" not in latest_row.columns:
            raise ValidationError("Hopsworks online vector missing event_time column 'dt'")
        dt_val = latest_row["dt"].iloc[0]
        if not (isinstance(dt_val, (int, np.integer, float)) and not math.isnan(dt_val)):
            raise ValidationError(f"Invalid non-numeric dt value in Hopsworks online row: {dt_val}")
        dt_int = int(dt_val)
        obs_datetime = datetime.fromtimestamp(dt_int, tz=timezone.utc)

        # Load canonical 114 schema
        schema_path = self.runtime_resolver.get_schema_path()
        import json
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
        canonical_features = schema_data.get("feature_names", [])
        if len(canonical_features) != 114:
            raise ValidationError(f"Expected 114 canonical features in schema, found {len(canonical_features)}")

        # Verify all 114 canonical features exist
        missing = [col for col in canonical_features if col not in latest_row.columns]
        if missing:
            raise ValidationError(f"Hopsworks online row missing canonical features: {missing[:5]}")

        # Construct single-row DataFrame in exact canonical schema order and validate finite values
        ordered_values = []
        for col in canonical_features:
            val = latest_row[col].iloc[0]
            if not isinstance(val, (int, float, np.number)) or math.isnan(val) or math.isinf(val):
                raise ValidationError(f"Invalid non-finite numeric value for feature '{col}': {val}")
            ordered_values.append(float(val))

        features_df = pd.DataFrame([ordered_values], columns=canonical_features)
        features_arr = np.array([ordered_values], dtype=np.float32)

        # Current AQI contract: use the physical current observation's epa_aqi
        if "epa_aqi" in latest_row.columns:
            current_aqi = float(latest_row["epa_aqi"].iloc[0])
        else:
            current_aqi = float(ordered_values[canonical_features.index("epa_aqi")])

        # Extract criteria pollutants
        pollutants: dict[str, float] = {}
        for pol in ["pm2_5", "pm10", "o3", "no2", "so2", "co", "nh3"]:
            if pol in latest_row.columns:
                pollutants[pol] = round(float(latest_row[pol].iloc[0]), 2)

        # Extract weather parameters
        weather: dict[str, float] = {}
        for w_col in ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "precipitation"]:
            if w_col in latest_row.columns:
                weather[w_col] = round(float(latest_row[w_col].iloc[0]), 2)

        # Dynamically derive dominant pollutant using standard US EPA sub-index calculation
        dominant_pollutant: str | None = None
        if pollutants:
            _, dominant_calc, _ = calculate_overall_aqi(pollutants)
            dominant_pollutant = dominant_calc

        return ResolvedFeatureObservation(
            features=features_df,
            features_array=features_arr,
            observed_at=obs_datetime,
            dt=dt_int,
            current_aqi=current_aqi,
            source="hopsworks",
            cloud_active=True,
            fallback_active=False,
            fallback_reason=None,
            dominant_pollutant=dominant_pollutant,
            pollutants=pollutants,
            weather=weather,
        )

    def _fetch_bootstrap(self) -> ResolvedFeatureObservation:
        """Load and validate the committed canonical bootstrap feature vector."""
        df_boot, metadata = self.runtime_resolver.load_bootstrap_feature_vector()
        features_arr = df_boot.values.astype(np.float32)
        obs_dt: datetime = metadata["input_observed_at"]
        dt_int = int(obs_dt.timestamp())

        pollutants = metadata.get("pollutants", {})
        weather = metadata.get("weather", {})

        # Derive or use verified metadata dominant pollutant
        dominant_pollutant: str | None = None
        if pollutants:
            _, dominant_calc, _ = calculate_overall_aqi(pollutants)
            dominant_pollutant = dominant_calc or metadata.get("dominant_pollutant")
        else:
            dominant_pollutant = metadata.get("dominant_pollutant")

        return ResolvedFeatureObservation(
            features=df_boot,
            features_array=features_arr,
            observed_at=obs_dt,
            dt=dt_int,
            current_aqi=float(metadata["current_aqi"]),
            source="bootstrap",
            cloud_active=False,
            fallback_active=False,
            fallback_reason=None,
            dominant_pollutant=dominant_pollutant,
            pollutants=pollutants,
            weather=weather,
        )

    def _resolve_uncached(self) -> ResolvedFeatureObservation:
        """Resolve feature observation based on active mode with fallback logic."""
        if self.mode == "bootstrap":
            logger.info("Resolving observation in strict 'bootstrap' mode.")
            return self._fetch_bootstrap()

        if self.mode == "hopsworks":
            logger.info("Resolving observation in strict 'hopsworks' mode.")
            return self._fetch_hopsworks()

        # mode == 'auto': prefer Hopsworks, fall back to bootstrap
        try:
            logger.info("Resolving observation in 'auto' mode: attempting Hopsworks online store...")
            obs = self._fetch_hopsworks()
            self._last_error = None
            return obs
        except Exception as e:
            logger.warning(
                f"Hopsworks online feature retrieval failed in 'auto' mode ({e}); activating bootstrap fallback."
            )
            self._last_error = f"{type(e).__name__}: {str(e)}"
            fallback_obs = self._fetch_bootstrap()
            fallback_obs.fallback_active = True
            fallback_obs.fallback_reason = self._last_error
            return fallback_obs

    def resolve_observation(self, force_refresh: bool = False) -> ResolvedFeatureObservation:
        """Resolve feature observation using thread-safe short-TTL caching.

        Args:
            force_refresh: If True, bypass in-memory cache and re-query the active source.

        Returns:
            ResolvedFeatureObservation instance.
        """
        with self._lock:
            now_mono = time.monotonic()
            if not force_refresh and self._cached_observation is not None:
                if (now_mono - self._cache_timestamp) < self.cache_ttl_seconds:
                    return self._cached_observation

            obs = self._resolve_uncached()
            self._cached_observation = obs
            self._cache_timestamp = now_mono
            self._last_source = obs.source
            return obs

    def get_status(self) -> dict[str, Any]:
        """Return non-blocking resolver operational metadata for health checks."""
        return {
            "mode": self.mode,
            "last_source": self._last_source,
            "last_error": self._last_error,
            "is_cached": self._cached_observation is not None,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }
