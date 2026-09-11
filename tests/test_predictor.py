"""Unit and integration tests for src.inference.predictor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.exceptions import ValidationError
from src.inference.cache import PredictionCache
from src.inference.model_loader import ModelLoader
from src.inference.post_processing import AQIPostProcessor
from src.inference.predictor import AQIPredictor


class TestAQIPredictor:
    """Test suite for AQIPredictor."""

    @pytest.fixture
    def predictor(self, tmp_path: Path) -> AQIPredictor:
        from src.inference.observation_resolver import FeatureObservationResolver
        loader = ModelLoader()
        processor = AQIPostProcessor()
        cache = PredictionCache(cache_file=tmp_path / "test_cache.json")
        resolver = FeatureObservationResolver(mode="bootstrap")
        return AQIPredictor(model_loader=loader, post_processor=processor, cache=cache, observation_resolver=resolver)




    @pytest.fixture
    def sample_feature_df(self) -> pd.DataFrame:
        loader = ModelLoader()
        schema = loader.load_schema()
        # Create a single-row DataFrame with dummy valid numeric data
        data = {col: [1.0] for col in schema}
        data["epa_aqi"] = [85.0]
        data["pm2_5"] = [35.0]
        data["temperature_2m"] = [22.5]
        data["relative_humidity_2m"] = [55.0]
        data["wind_speed_10m"] = [3.5]
        data["dt"] = [1672531199]
        return pd.DataFrame(data)

    def test_predict_72h_from_dataframe_returns_complete_contract(
        self, predictor: AQIPredictor, sample_feature_df: pd.DataFrame
    ):
        result = predictor.predict_72h(
            features=sample_feature_df,
            current_aqi=85.0,
            forecast_origin=datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc),
            input_observed_at=datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Check provenance keys
        assert result["model_id"] == "EXP-019"
        assert result["model_version"] == "1.0-production"
        assert result["feature_count"] == 114
        assert result["data_status"] in ["live", "stale"]
        assert "generated_at" in result
        assert result["current_aqi"] == 85.0
        assert result["current_category"] == "Moderate"
        assert "inference_latency_ms" in result
        assert isinstance(result["inference_latency_ms"], float)

        # Check forecast points
        forecasts = result["forecasts"]
        assert len(forecasts) == 72
        assert forecasts[0]["horizon"] == 1
        assert forecasts[71]["horizon"] == 72
        for f in forecasts:
            assert f["aqi"] >= 0.0
            assert f["error_lower"] <= f["error_upper"]
            assert isinstance(f["high_severity"], bool)
            assert isinstance(f["hazardous"], bool)

        # Check summary
        summary = result["summary"]
        assert "peak_aqi" in summary
        assert 1 <= summary["peak_horizon"] <= 72

    def test_forecast_origin_equals_input_observation_time(
        self, predictor: AQIPredictor, sample_feature_df: pd.DataFrame
    ):
        """forecast_origin must strictly match input observation time, not current time."""
        obs_time = datetime(2026, 8, 31, 7, 0, 0, tzinfo=timezone.utc)
        result = predictor.predict_72h(
            features=sample_feature_df,
            input_observed_at=obs_time,
        )
        assert result["input_observed_at"] == "2026-08-31T07:00:00+00:00"
        assert result["forecast_origin"] == "2026-08-31T07:00:00+00:00"

    def test_h1_time_is_origin_plus_one_hour(
        self, predictor: AQIPredictor, sample_feature_df: pd.DataFrame
    ):
        obs_time = datetime(2026, 8, 31, 7, 0, 0, tzinfo=timezone.utc)
        result = predictor.predict_72h(
            features=sample_feature_df,
            input_observed_at=obs_time,
        )
        f1 = result["forecasts"][0]
        assert f1["horizon"] == 1
        assert f1["forecast_time"] == "2026-08-31T08:00:00+00:00"

    def test_h72_time_is_origin_plus_72_hours(
        self, predictor: AQIPredictor, sample_feature_df: pd.DataFrame
    ):
        obs_time = datetime(2026, 8, 31, 7, 0, 0, tzinfo=timezone.utc)
        result = predictor.predict_72h(
            features=sample_feature_df,
            input_observed_at=obs_time,
        )
        f72 = result["forecasts"][71]
        assert f72["horizon"] == 72
        assert f72["forecast_time"] == "2026-09-03T07:00:00+00:00"

    def test_generated_at_does_not_shift_forecast_horizons(
        self, predictor: AQIPredictor, sample_feature_df: pd.DataFrame
    ):
        obs_time = datetime(2026, 8, 31, 7, 0, 0, tzinfo=timezone.utc)
        result = predictor.predict_72h(
            features=sample_feature_df,
            input_observed_at=obs_time,
        )
        gen_time = datetime.fromisoformat(result["generated_at"])
        assert gen_time.year == 2026
        # Horizons are anchored to August 31, not the generation time in September
        assert result["forecasts"][0]["forecast_time"].startswith("2026-08-31T08:00:00")

    def test_predict_72h_from_numpy_array(
        self, predictor: AQIPredictor
    ):
        loader = ModelLoader()
        n_feat = len(loader.load_schema())
        X = np.ones((1, n_feat), dtype=np.float32)
        result = predictor.predict_72h(features=X, current_aqi=120.0)

        assert result["feature_count"] == 114
        assert len(result["forecasts"]) == 72
        assert result["current_aqi"] == 120.0

    def test_predict_72h_enforces_exact_column_order(
        self, predictor: AQIPredictor, sample_feature_df: pd.DataFrame
    ):
        """Scrambled column order must raise ValidationError."""
        loader = ModelLoader()
        schema = loader.load_schema()
        scrambled_cols = list(schema)
        # Swap columns 0 and 1
        scrambled_cols[0], scrambled_cols[1] = scrambled_cols[1], scrambled_cols[0]
        df_bad = sample_feature_df[scrambled_cols]

        with pytest.raises(ValidationError, match="order does not match"):
            predictor.predict_72h(features=df_bad)

    def test_staleness_flag_triggers_when_data_is_old(
        self, predictor: AQIPredictor, sample_feature_df: pd.DataFrame
    ):
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(hours=5)  # 5 hours old (> 3h threshold)

        result = predictor.predict_72h(
            features=sample_feature_df,
            input_observed_at=old_time,
            forecast_origin=old_time,
        )

        assert result["is_stale"] is True
        assert result["data_status"] == "stale"
        assert result["input_age_hours"] >= 4.9

    def test_predict_latest_executes_on_real_feature_dataset(
        self, predictor: AQIPredictor
    ):
        result = predictor.predict_latest(use_cache=False)
        assert result["model_id"] == "EXP-019"
        assert len(result["forecasts"]) == 72
        assert result["current_aqi"] > 0.0
        assert "input_observed_at" in result
        assert result["input_observed_at"] == "2026-08-31T07:00:00+00:00"
        assert result["forecast_origin"] == "2026-08-31T07:00:00+00:00"
        assert result["forecasts"][0]["forecast_time"] == "2026-08-31T08:00:00+00:00"
        assert result["forecasts"][71]["forecast_time"] == "2026-09-03T07:00:00+00:00"

    def test_force_refresh_does_not_mark_stale_data_fresh(
        self, predictor: AQIPredictor
    ):
        """Bypassing cache to recompute from historical dataset must still report stale."""
        result = predictor.predict_latest(use_cache=True, force_refresh=True)
        assert result["is_stale"] is True
        assert result["data_status"] == "stale"
        assert result["input_age_hours"] > 3.0

    def test_predict_latest_cache_integration(
        self, predictor: AQIPredictor
    ):
        res1 = predictor.predict_latest(use_cache=True, force_refresh=True)
        res2 = predictor.predict_latest(use_cache=True, force_refresh=False)
        # Dynamic freshness timestamp updates per request
        assert res1["forecasts"] == res2["forecasts"]
        assert res1["current_aqi"] == res2["current_aqi"]
        assert res1["forecast_origin"] == res2["forecast_origin"]
        assert res1["input_observed_at"] == res2["input_observed_at"]
        assert res1["observation_dt"] == res2["observation_dt"]
        assert res1["feature_source"] == res2["feature_source"]
        assert "generated_at" in res1 and "generated_at" in res2

    def test_get_latest_observation_does_not_require_model_inference(
        self, predictor: AQIPredictor
    ):
        """Reading current observation reads telemetry and categorizes without running model."""
        obs = predictor.get_latest_observation()
        assert "current_aqi" in obs
        assert "dominant_pollutant" in obs
        assert "category" in obs
        assert "color" in obs
        assert "pollutants" in obs
        assert "weather" in obs
        assert obs["data_status"] in ["live", "stale"]
        assert "input_observed_at" in obs
        assert "retrieved_at" in obs

    def test_get_model_metadata(self, predictor: AQIPredictor):
        meta = predictor.get_model_metadata()
        assert meta["model_id"] == "EXP-019"
        assert meta["feature_count"] == 114
        assert meta["status"] == "validated_production_champion"
        assert "test_benchmark_metrics" in meta
        assert meta["test_benchmark_metrics"]["overall_rmse"] == 75.91
