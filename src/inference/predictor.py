"""Pearls AQI Predictor - Multi-Horizon AQI Inference Engine.

Orchestrates model execution, input schema verification, data freshness auditing,
post-processing enrichment, and cache coordination for 72-hour AQI forecasts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.exceptions import ValidationError
from src.inference.alerting import evaluate_current_alert, evaluate_forecast_alerts
from src.inference.cache import PredictionCache
from src.inference.explainer import ModelExplainer
from src.inference.model_loader import ModelLoader
from src.inference.observation_resolver import FeatureObservationResolver, ResolvedFeatureObservation
from src.inference.post_processing import AQIPostProcessor
from src.inference.runtime_resolver import RuntimeAssetResolver
from src.logger import logger


class AQIPredictor:
    """Production 72-hour multi-horizon AQI prediction service."""

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        post_processor: AQIPostProcessor | None = None,
        cache: PredictionCache | None = None,
        runtime_resolver: RuntimeAssetResolver | None = None,
        observation_resolver: FeatureObservationResolver | None = None,
    ) -> None:
        """Initialize AQIPredictor.

        Args:
            model_loader: ModelLoader instance (default creates standard loader).
            post_processor: AQIPostProcessor instance (default creates standard processor).
            cache: PredictionCache instance (default creates standard cache).
            runtime_resolver: Optional RuntimeAssetResolver instance.
            observation_resolver: Optional FeatureObservationResolver instance.
        """
        self.runtime_resolver = runtime_resolver or RuntimeAssetResolver()
        self.model_loader = model_loader or ModelLoader(resolver=self.runtime_resolver)
        self.post_processor = post_processor or AQIPostProcessor(resolver=self.runtime_resolver)
        self.cache = cache or PredictionCache()
        self.observation_resolver = observation_resolver or FeatureObservationResolver(
            runtime_resolver=self.runtime_resolver
        )

    def predict_72h(
        self,
        features: pd.DataFrame | np.ndarray,
        current_aqi: float | None = None,
        forecast_origin: datetime | None = None,
        input_observed_at: datetime | None = None,
        fallback_active: bool = False,
        feature_source: str = "unknown",
    ) -> dict[str, Any]:
        """Generate a 72-hour forecast from a single input feature vector.

        Args:
            features: DataFrame or 2D array of shape (1, 114) containing features matching canonical schema.
            current_aqi: Optional unscaled current AQI value. If not provided, extracted from input features.
            forecast_origin: Timestamp of forecast origin (default: current UTC time).
            input_observed_at: Timestamp when input telemetry was observed (for staleness auditing).
            fallback_active: Whether bootstrap fallback is currently active.
            feature_source: Name of feature provider ('hopsworks', 'bootstrap', etc.).

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
        # 5. Freshness & Staleness Metadata
        input_age_hours = round(max(0.0, (now - observed_time).total_seconds() / 3600.0), 2)
        is_stale = bool(input_age_hours > 3.0)
        data_status = "stale" if is_stale else "live"

        summary = self.post_processor.build_summary(
            forecasts,
            data_is_stale=is_stale,
            fallback_active=fallback_active,
            feature_source=feature_source,
        )
        forecast_alert = summary.get("forecast_alert", {})

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
            "forecast_alert": forecast_alert,
            "forecasts": forecasts,
        }

    def predict_latest(
        self,
        use_cache: bool = True,
        force_refresh: bool = False,
        dataset_path: Path | str | None = None,
        observation: ResolvedFeatureObservation | None = None,
    ) -> dict[str, Any]:
        """Generate forecast from the latest available telemetry row or bootstrap feature vector.

        Args:
            use_cache: Whether to return cached predictions if valid (default: True).
            force_refresh: Whether to bypass cache and recompute from latest stored features (default: False).
            dataset_path: Optional path to features CSV.
            observation: Optional pre-resolved ResolvedFeatureObservation.

        Returns:
            Dictionary matching the PredictionResult contract.
        """
        cache_key = "latest_72h_forecast"
        if use_cache and not force_refresh and observation is None and dataset_path is None:
            cached_result = self.cache.get(cache_key, ttl_seconds=60)
            if cached_result is not None:
                # Dynamically calculate freshness on each response
                now = datetime.now(timezone.utc)
                obs_time = datetime.fromisoformat(cached_result["input_observed_at"])
                age_h = round(max(0.0, (now - obs_time).total_seconds() / 3600.0), 2)
                stale = age_h > 3.0
                cached_result["input_age_hours"] = age_h
                cached_result["is_stale"] = stale
                cached_result["data_status"] = "stale" if stale else "live"
                cached_result["generated_at"] = now.isoformat()
                if "forecast_alert" in cached_result and isinstance(cached_result["forecast_alert"], dict):
                    cached_result["forecast_alert"]["data_is_stale"] = stale
                if "summary" in cached_result and isinstance(cached_result["summary"], dict):
                    if "forecast_alert" in cached_result["summary"] and isinstance(cached_result["summary"]["forecast_alert"], dict):
                        cached_result["summary"]["forecast_alert"]["data_is_stale"] = stale
                return cached_result

        path = Path(dataset_path) if dataset_path else None
        if path and path.exists():
            logger.info(f"Loading latest telemetry from {path}...")
            df_latest = pd.read_csv(path).tail(1).copy()
            if df_latest.empty:
                raise ValidationError(f"No records found in {path}")

            ts_val = pd.to_datetime(df_latest["datetime_utc"].iloc[0], utc=True) if "datetime_utc" in df_latest.columns else datetime.now(timezone.utc)
            input_observed_at = ts_val.to_pydatetime()
            current_aqi = float(df_latest["epa_aqi"].iloc[0]) if "epa_aqi" in df_latest.columns else 0.0
            expected_schema = self.model_loader.load_schema()
            df_features = df_latest[expected_schema].copy()
            obs_dt = int(df_latest["dt"].iloc[0]) if "dt" in df_latest.columns else int(input_observed_at.timestamp())
            source_name = "csv"
            cloud_active = False
            fallback_active = False
            fallback_reason = None
        else:
            obs = observation or self.observation_resolver.resolve_observation(force_refresh=force_refresh)
            df_features = obs.features
            current_aqi = obs.current_aqi
            input_observed_at = obs.observed_at
            obs_dt = obs.dt
            source_name = obs.source
            cloud_active = obs.cloud_active
            fallback_active = obs.fallback_active
            fallback_reason = obs.fallback_reason

        result = self.predict_72h(
            features=df_features,
            current_aqi=current_aqi,
            forecast_origin=input_observed_at,
            input_observed_at=input_observed_at,
            fallback_active=fallback_active,
            feature_source=source_name,
        )

        result["feature_source"] = source_name
        result["cloud_active"] = cloud_active
        result["fallback_active"] = fallback_active
        result["fallback_reason"] = fallback_reason
        result["observation_dt"] = obs_dt

        if use_cache and dataset_path is None:
            self.cache.set(cache_key, result)

        return result

    def get_latest_observation(
        self,
        dataset_path: Path | str | None = None,
        force_refresh: bool = False,
        observation: ResolvedFeatureObservation | None = None,
    ) -> dict[str, Any]:
        """Read and categorize the latest telemetry observation without model inference.

        Args:
            dataset_path: Optional path to features CSV.
            force_refresh: Whether to bypass resolver cache.
            observation: Optional pre-resolved ResolvedFeatureObservation.

        Returns:
            Dictionary containing latest observation values, EPA category, and freshness metadata.
        """
        path = Path(dataset_path) if dataset_path else None
        now = datetime.now(timezone.utc)

        if path and path.exists():
            df_latest = pd.read_csv(path).tail(1).copy()
            if df_latest.empty:
                raise ValidationError(f"No records found in {path}")

            ts_val = pd.to_datetime(df_latest["datetime_utc"].iloc[0], utc=True) if "datetime_utc" in df_latest.columns else now
            observed_time = ts_val.to_pydatetime()
            current_aqi = float(df_latest["epa_aqi"].iloc[0]) if "epa_aqi" in df_latest.columns else 0.0
            dominant = str(df_latest["dominant_pollutant"].iloc[0]) if "dominant_pollutant" in df_latest.columns else None
            pollutants = {pol: round(float(df_latest[pol].iloc[0]), 2) for pol in ["pm2_5", "pm10", "o3", "no2", "so2", "co"] if pol in df_latest.columns}
            weather = {w_col: round(float(df_latest[w_col].iloc[0]), 2) for w_col in ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"] if w_col in df_latest.columns}
            obs_dt = int(df_latest["dt"].iloc[0]) if "dt" in df_latest.columns else int(observed_time.timestamp())
            source_name = "csv"
            cloud_active = False
            fallback_active = False
            fallback_reason = None
            age_hours = round(max(0.0, (now - observed_time).total_seconds() / 3600.0), 2)
            stale = age_hours > 3.0
        else:
            obs = observation or self.observation_resolver.resolve_observation(force_refresh=force_refresh)
            observed_time = obs.observed_at
            current_aqi = obs.current_aqi
            dominant = obs.dominant_pollutant
            pollutants = obs.pollutants
            weather = obs.weather
            obs_dt = obs.dt
            source_name = obs.source
            cloud_active = obs.cloud_active
            fallback_active = obs.fallback_active
            fallback_reason = obs.fallback_reason
            age_hours = obs.get_age_hours(now)
            stale = obs.is_stale(now)

        data_status = "stale" if stale else "live"
        point = self.post_processor.process_horizon_point(0, observed_time, current_aqi)
        current_alert = evaluate_current_alert(
            aqi=current_aqi,
            data_is_stale=stale,
            feature_source=source_name,
            fallback_active=fallback_active,
        )

        return {
            "data_status": data_status,
            "input_observed_at": observed_time.isoformat(),
            "observation_dt": obs_dt,
            "retrieved_at": now.isoformat(),
            "input_age_hours": age_hours,
            "is_stale": stale,
            "current_aqi": round(current_aqi, 1),
            "dominant_pollutant": dominant,
            "category": point["category"],
            "color": point["color"],
            "health_advisory": point["health_advisory"],
            "alert_level": point["alert_level"],
            "severity_rank": point["severity_rank"],
            "alert": current_alert.to_dict(),
            "pollutants": pollutants,
            "weather": weather,
            "feature_source": source_name,
            "cloud_active": cloud_active,
            "fallback_active": fallback_active,
            "fallback_reason": fallback_reason,
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
            "walk_forward_cv_metrics": {
                "folds": 4,
                "embargo_gap_hours": 72,
                "mean_rmse": 83.44,
                "fold_std_rmse": 4.9,
                "naive_mean_rmse": 110.46,
                "relative_gain_pct": 24.46,
            },
            "status": "validated_production_champion",
        }

    def get_global_explainability(self) -> dict[str, Any]:
        """Return precomputed global feature importance rankings and manifest."""
        global_path = self.runtime_resolver.get_explainability_dir() / "global_shap_importance.json"
        if not global_path.exists():
            from src.inference.build_explainability_artifacts import build_artifacts
            build_artifacts()

        with open(global_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def explain_latest(
        self,
        horizon: int = 24,
        top_k: int = 10,
        dataset_path: Path | str | None = None,
        observation: ResolvedFeatureObservation | None = None,
    ) -> dict[str, Any]:
        """Generate SHAP feature attribution and persistence decomposition for the latest observation.

        Args:
            horizon: Target horizon index (1..72).
            top_k: Number of top features to return.
            dataset_path: Optional path to features dataset.
            observation: Optional pre-resolved ResolvedFeatureObservation.

        Returns:
            Dictionary matching authoritative explanation contract.
        """
        now = datetime.now(timezone.utc)
        path = Path(dataset_path) if dataset_path else None
        schema = self.model_loader.load_schema()

        if path and path.exists():
            df_latest = pd.read_csv(path).tail(1).copy()
            if df_latest.empty:
                raise ValidationError(f"No records found in {path}")
            ts_val = pd.to_datetime(df_latest["datetime_utc"].iloc[0], utc=True) if "datetime_utc" in df_latest.columns else now
            input_observed_at = ts_val.to_pydatetime()
            current_aqi_val = float(df_latest["epa_aqi"].iloc[0]) if "epa_aqi" in df_latest.columns else 0.0
            raw_vec = df_latest[schema].values
            obs_dt = int(df_latest["dt"].iloc[0]) if "dt" in df_latest.columns else int(input_observed_at.timestamp())
            source_name = "csv"
            cloud_active = False
            fallback_active = False
            fallback_reason = None
            age_hours = round(max(0.0, (now - input_observed_at).total_seconds() / 3600.0), 2)
            stale = age_hours > 3.0
            data_status = "stale" if stale else "live"
        else:
            obs = observation or self.observation_resolver.resolve_observation()
            raw_vec = obs.features_array
            current_aqi_val = obs.current_aqi
            input_observed_at = obs.observed_at
            obs_dt = obs.dt
            source_name = obs.source
            cloud_active = obs.cloud_active
            fallback_active = obs.fallback_active
            fallback_reason = obs.fallback_reason
            age_hours = obs.get_age_hours(now)
            stale = obs.is_stale(now)
            data_status = "stale" if stale else "live"

        # Initialize explainer with loaded production model and scaler
        model = self.model_loader.load_model()
        scaler = self.model_loader.load_scaler()
        explainer = ModelExplainer(
            model=model,
            scaler=scaler,
            feature_names=schema,
            resolver=self.runtime_resolver,
        )

        # Compute explanation passing the exact resolved current_aqi
        explanation = explainer.explain_horizon(
            raw_feature_vector=raw_vec,
            horizon=horizon,
            top_k=top_k,
            current_aqi=current_aqi_val,
        )

        # Attach timeline provenance and freshness metadata matching /api/forecast
        target_time = input_observed_at + timedelta(hours=horizon)
        global_info = self.get_global_explainability()

        return {
            "model_id": "EXP-019",
            "model_version": "1.0-production",
            "horizon": horizon,
            "data_status": data_status,
            "input_observed_at": input_observed_at.isoformat(),
            "forecast_origin": input_observed_at.isoformat(),
            "observation_dt": obs_dt,
            "target_time": target_time.isoformat(),
            "generated_at": now.isoformat(),
            "input_age_hours": age_hours,
            "is_stale": stale,
            "feature_source": source_name,
            "cloud_active": cloud_active,
            "fallback_active": fallback_active,
            "fallback_reason": fallback_reason,
            **explanation,
            "global_persistence_mean_contribution": global_info.get("global_persistence_mean_contribution", 0.0),
            "global_top_features": global_info.get("overall_feature_importance", [])[:top_k],
        }


