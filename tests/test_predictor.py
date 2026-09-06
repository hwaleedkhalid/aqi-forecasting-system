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
        loader = ModelLoader()
        processor = AQIPostProcessor()
        cache = PredictionCache(cache_file=tmp_path / "test_cache.json")
        return AQIPredictor(model_loader=loader, post_processor=processor, cache=cache)

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
        assert result["is_stale"] is False
        assert result["current_aqi"] == 85.0
        assert result["current_category"] == "Moderate"
        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], float)

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
            forecast_origin=now,
        )

        assert result["is_stale"] is True
        assert result["input_age_hours"] >= 4.9

    def test_predict_latest_executes_on_real_feature_dataset(
        self, predictor: AQIPredictor
    ):
        result = predictor.predict_latest(use_cache=False)
        assert result["model_id"] == "EXP-019"
        assert len(result["forecasts"]) == 72
        assert result["current_aqi"] > 0.0
        assert "input_observed_at" in result

    def test_predict_latest_cache_integration(
        self, predictor: AQIPredictor
    ):
        # First call: cache miss, computes
        res1 = predictor.predict_latest(use_cache=True, force_refresh=True)
        # Second call: cache hit
        res2 = predictor.predict_latest(use_cache=True, force_refresh=False)
        assert res1 == res2

    def test_get_model_metadata(self, predictor: AQIPredictor):
        meta = predictor.get_model_metadata()
        assert meta["model_id"] == "EXP-019"
        assert meta["feature_count"] == 114
        assert meta["status"] == "validated_production_champion"
        assert "test_benchmark_metrics" in meta
        assert meta["test_benchmark_metrics"]["overall_rmse"] == 75.91
