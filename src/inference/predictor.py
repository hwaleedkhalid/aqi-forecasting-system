"""Pearls AQI Predictor - Multi-Horizon AQI Inference Engine.

Orchestrates model execution, input schema verification, data freshness auditing,
post-processing enrichment, and cache coordination for 72-hour AQI forecasts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.exceptions import ValidationError
from src.logger import logger
from src.inference.cache import PredictionCache
from src.inference.model_loader import ModelLoader
from src.inference.post_processing import AQIPostProcessor


class AQIPredictor:
    """Production 72-hour multi-horizon AQI prediction service."""

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        post_processor: AQIPostProcessor | None = None,
        cache: PredictionCache | None = None,
    ) -> None:
        """Initialize AQIPredictor.

        Args:
            model_loader: ModelLoader instance (default creates standard loader).
            post_processor: AQIPostProcessor instance (default creates standard processor).
            cache: PredictionCache instance (default creates standard cache).
        """
        self.model_loader = model_loader or ModelLoader()
        self.post_processor = post_processor or AQIPostProcessor()
        self.cache = cache or PredictionCache()

    def predict_72h(
        self,
        features: pd.DataFrame | np.ndarray,
        current_aqi: float | None = None,
        forecast_origin: datetime | None = None,
        input_observed_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Generate a 72-hour forecast from a single input feature vector.

        Args:
            features: DataFrame or 2D array of shape (1, 114) containing features matching canonical schema.
            current_aqi: Optional unscaled current AQI value. If not provided, extracted from input features.
            forecast_origin: Timestamp of forecast origin (default: current UTC time).
            input_observed_at: Timestamp when input telemetry was observed (for staleness auditing).

        Returns:
            Dictionary matching the PredictionResult contract.
        """
        start_time = time.perf_counter()

        now = datetime.now(timezone.utc)
        observed_time = input_observed_at or forecast_origin or now
        if observed_time.tzinfo is None:
            observed_time = observed_time.replace(tzinfo=timezone.utc)

        # Forecast origin is strictly the observation time!
        origin = forecast_origin or observed_time
        if origin.tzinfo is None:
            origin = origin.replace(tzinfo=timezone.utc)

        expected_schema = self.model_loader.load_schema()
        scaler = self.model_loader.load_scaler()
        model = self.model_loader.load_model()

        # 1. Feature Extraction & Column Order Validation
        curr_aqi_val: float
        if isinstance(features, pd.DataFrame):
            self.model_loader.validate_feature_vector(features)
            # Reindex to guarantee exact column order matching schema
            X_df = features[expected_schema].copy()

            if current_aqi is not None:
                curr_aqi_val = float(current_aqi)
            elif "epa_aqi" in features.columns:
                curr_aqi_val = float(features["epa_aqi"].iloc[0])
            elif "epa_aqi_lag_1h" in features.columns:
                curr_aqi_val = float(features["epa_aqi_lag_1h"].iloc[0])
            else:
                curr_aqi_val = float(X_df.iloc[0, model.current_aqi_col_idx])

            X_raw = X_df.values.astype(np.float32)
        elif isinstance(features, np.ndarray):
            X_raw = np.asarray(features, dtype=np.float32)
            if X_raw.ndim == 1:
                X_raw = X_raw.reshape(1, -1)
            if X_raw.shape[1] != len(expected_schema):
                raise ValidationError(
                    f"Feature vector length mismatch: expected {len(expected_schema)}, got {X_raw.shape[1]}"
                )
            if current_aqi is not None:
                curr_aqi_val = float(current_aqi)
            else:
                curr_aqi_val = float(X_raw[0, model.current_aqi_col_idx])
        else:
            raise ValidationError(f"Unsupported features type: {type(features).__name__}")

        # 2. Scaling (scaler expects 114 features in exact canonical order)
        X_scaled = scaler.transform(X_raw)

        # 3. Model Inference (unscaled current_aqi passed for persistence blending, clip_max=None preserves extremes)
        curr_aqi_arr = np.array([curr_aqi_val], dtype=np.float32)
        raw_preds = model.predict(X_scaled, current_aqi=curr_aqi_arr, clip_max=None)
        raw_preds_1d = raw_preds.flatten()

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # 4. Post-processing & Enrichment
        forecasts = self.post_processor.process_72h_forecast(raw_preds_1d, forecast_origin=origin)
        summary = self.post_processor.build_summary(forecasts)

        # 5. Freshness & Staleness Metadata
        input_age_hours = round(max(0.0, (now - observed_time).total_seconds() / 3600.0), 2)
        is_stale = bool(input_age_hours > 3.0)
        data_status = "stale" if is_stale else "live"

        curr_cat, curr_color = self.post_processor.process_horizon_point(
            0, origin, curr_aqi_val
        )["category"], self.post_processor.process_horizon_point(0, origin, curr_aqi_val)["color"]

        return {
            "data_status": data_status,
            "input_observed_at": observed_time.isoformat(),
            "forecast_origin": origin.isoformat(),
            "generated_at": now.isoformat(),
            "input_age_hours": input_age_hours,
            "is_stale": is_stale,
            "model_id": "EXP-019",
            "model_version": "1.0-production",
            "feature_schema_version": "v2_weather_enriched",
            "feature_count": len(expected_schema),
            "inference_latency_ms": elapsed_ms,
            "current_aqi": round(curr_aqi_val, 1),
            "current_category": curr_cat,
            "current_color": curr_color,
            "summary": summary,
            "forecasts": forecasts,
        }

    def predict_latest(
        self,
        use_cache: bool = True,
        force_refresh: bool = False,
        dataset_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Generate forecast from the latest available telemetry row.

        Args:
            use_cache: Whether to return cached predictions if valid (default: True).
            force_refresh: Whether to bypass cache and recompute from latest stored features (default: False).
            dataset_path: Path to features CSV (default: data/processed/features_v2_weather.csv).

        Returns:
            Dictionary matching the PredictionResult contract.
        """
        cache_key = "latest_72h_forecast"
        if use_cache and not force_refresh:
            cached_result = self.cache.get(cache_key, ttl_seconds=3600)
            if cached_result is not None:
                return cached_result

        path = Path(dataset_path or PROCESSED_DATA_DIR / "features_v2_weather.csv")
        if not path.exists():
            raise FileNotFoundError(f"Feature dataset not found at {path}")

        # Read the latest rows to construct the input
        logger.info(f"Loading latest telemetry from {path}...")
        df_latest = pd.read_csv(path).tail(1).copy()
        if df_latest.empty:
            raise ValidationError(f"No records found in {path}")

        # Extract timestamp metadata
        input_observed_at: datetime | None = None
        if "datetime_utc" in df_latest.columns:
            ts_val = pd.to_datetime(df_latest["datetime_utc"].iloc[0], utc=True)
            input_observed_at = ts_val.to_pydatetime()

        current_aqi = None
        if "epa_aqi" in df_latest.columns:
            current_aqi = float(df_latest["epa_aqi"].iloc[0])

        expected_schema = self.model_loader.load_schema()
        df_features = df_latest[expected_schema].copy()

        result = self.predict_72h(
            features=df_features,
            current_aqi=current_aqi,
            forecast_origin=input_observed_at,
            input_observed_at=input_observed_at,
        )

        if use_cache:
            self.cache.set(cache_key, result)

        return result

    def get_latest_observation(
        self,
        dataset_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Read and categorize the latest telemetry observation without model inference.

        Args:
            dataset_path: Path to features CSV (default: data/processed/features_v2_weather.csv).

        Returns:
            Dictionary containing latest observation values, EPA category, and freshness metadata.
        """
        path = Path(dataset_path or PROCESSED_DATA_DIR / "features_v2_weather.csv")
        if not path.exists():
            raise FileNotFoundError(f"Feature dataset not found at {path}")

        df_latest = pd.read_csv(path).tail(1).copy()
        if df_latest.empty:
            raise ValidationError(f"No records found in {path}")

        now = datetime.now(timezone.utc)
        observed_time = now
        if "datetime_utc" in df_latest.columns:
            ts_val = pd.to_datetime(df_latest["datetime_utc"].iloc[0], utc=True)
            observed_time = ts_val.to_pydatetime()

        input_age_hours = round(max(0.0, (now - observed_time).total_seconds() / 3600.0), 2)
        is_stale = bool(input_age_hours > 3.0)
        data_status = "stale" if is_stale else "live"

        current_aqi = float(df_latest["epa_aqi"].iloc[0]) if "epa_aqi" in df_latest.columns else 0.0
        dominant = str(df_latest["dominant_pollutant"].iloc[0]) if "dominant_pollutant" in df_latest.columns else "pm2_5"

        point = self.post_processor.process_horizon_point(0, observed_time, current_aqi)

        pollutants = {}
        for pol in ["pm2_5", "pm10", "o3", "no2", "so2", "co"]:
            if pol in df_latest.columns:
                pollutants[pol] = round(float(df_latest[pol].iloc[0]), 2)

        weather = {}
        for w_col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]:
            if w_col in df_latest.columns:
                weather[w_col] = round(float(df_latest[w_col].iloc[0]), 2)

        return {
            "data_status": data_status,
            "input_observed_at": observed_time.isoformat(),
            "retrieved_at": now.isoformat(),
            "input_age_hours": input_age_hours,
            "is_stale": is_stale,
            "current_aqi": round(current_aqi, 1),
            "dominant_pollutant": dominant,
            "category": point["category"],
            "color": point["color"],
            "health_advisory": point["health_advisory"],
            "pollutants": pollutants,
            "weather": weather,
        }

    def get_model_metadata(self) -> dict[str, Any]:
        """Return production model provenance, architecture specification, and benchmark metrics."""
        return {
            "model_id": "EXP-019",
            "model_name": "PersistenceAwareHybridModel",
            "version": "1.0-production",
            "architecture": "LightGBM (h1-6) + Ridge (h7-37) + Persistence Blended Ridge (h38-72)",
            "feature_schema_version": "v2_weather_enriched",
            "feature_count": 114,
            "training_dataset_span": "2020-11-28 to 2025-06-21 UTC",
            "test_benchmark_metrics": {
                "partition": "Held-out Untouched Out-of-Time Test Set (N=9,311)",
                "overall_rmse": 75.91,
                "overall_mae": 53.55,
                "overall_r2": 0.4858,
                "h1_rmse": 50.43,
                "h72_rmse": 77.43,
                "hazardous_gt300_rmse": 142.65,
            },
            "walk_forward_stability": {
                "folds_evaluated": 4,
                "folds_won_vs_naive": "4/4",
                "mean_rmse": 83.44,
                "naive_mean_rmse": 110.46,
                "relative_gain_pct": 24.46,
            },
            "status": "validated_production_champion",
        }
