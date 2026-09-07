"""Tests for Phase B RuntimeAssetResolver, Clean-Clone guarantees, and Frontend Isolation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from src.api.app import create_app
from src.dashboard.data_client import DashboardDataClient
from src.exceptions import ValidationError
from src.inference.runtime_resolver import RuntimeAssetResolver, compute_file_sha256


class TestRuntimeAssetResolver:
    """Test suite for RuntimeAssetResolver and manifest SHA256 verification."""

    def test_resolver_production_paths_exist_and_verified(self):
        resolver = RuntimeAssetResolver()
        model_path = resolver.get_model_path()
        scaler_path = resolver.get_scaler_path()
        schema_path = resolver.get_schema_path()
        error_path = resolver.get_error_intervals_path()
        manifest_path = resolver.get_production_manifest_path()

        assert model_path.is_file()
        assert scaler_path.is_file()
        assert schema_path.is_file()
        assert error_path.is_file()
        assert manifest_path.is_file()

        manifest = resolver.verify_runtime_integrity()
        assert manifest["model_id"] == "EXP-019"
        assert manifest["feature_count"] == 114

    def test_resolver_explainability_paths_exist_and_verified(self):
        resolver = RuntimeAssetResolver()
        expl_dir = resolver.get_explainability_dir()
        bg_path = expl_dir / "shap_background.npy"
        global_path = expl_dir / "global_shap_importance.json"
        manifest_path = expl_dir / "explainer_manifest.json"

        assert bg_path.is_file()
        assert global_path.is_file()
        assert manifest_path.is_file()

        expl_manifest = resolver.verify_explainability_integrity()
        assert "production_model_sha256" in expl_manifest
        assert "shap_background_sha256" in expl_manifest

    def test_resolver_bootstrap_vector_exists_and_valid(self):
        resolver = RuntimeAssetResolver()
        df, meta = resolver.load_bootstrap_feature_vector()
        assert df.shape == (1, 114)
        assert meta["current_aqi"] == 100.0
        assert meta["source"] == "bootstrap"
        assert "pollutants" in meta
        assert "weather" in meta

    def test_resolver_tampered_model_fails_closed(self, tmp_path):
        runtime_dir = tmp_path / "runtime"
        prod_dir = runtime_dir / "production"
        prod_dir.mkdir(parents=True)

        real_resolver = RuntimeAssetResolver()
        shutil.copy(real_resolver.get_model_path(), prod_dir / "production_hybrid_model.joblib")
        shutil.copy(real_resolver.get_scaler_path(), prod_dir / "feature_scaler_v2_weather.joblib")
        shutil.copy(real_resolver.get_schema_path(), prod_dir / "feature_schema_v2_weather.json")
        shutil.copy(real_resolver.get_error_intervals_path(), prod_dir / "empirical_error_intervals.json")
        shutil.copy(real_resolver.get_production_manifest_path(), prod_dir / "manifest.json")

        # Corrupt model file
        with open(prod_dir / "production_hybrid_model.joblib", "ab") as f:
            f.write(b"CORRUPTION_BYTES")

        tampered_resolver = RuntimeAssetResolver(mode="runtime", base_runtime_dir=runtime_dir)
        with pytest.raises(ValidationError) as exc_info:
            tampered_resolver.verify_runtime_integrity()
        assert "Integrity violation" in str(exc_info.value) or "SHA256" in str(exc_info.value)

    def test_api_health_check_returns_healthy(self):
        app = create_app({"TESTING": True})
        client = app.test_client()
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["service_ready"] is True
        assert data["model_loaded"] is True
        assert data["model_id"] == "EXP-019"


class TestCleanBootstrapInferenceParity:
    """Test suite ensuring clean bootstrap inference functions without historical CSVs."""

    def test_api_current_endpoint_bootstrap_parity(self):
        app = create_app({"TESTING": True})
        client = app.test_client()
        resp = client.get("/api/current")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["current_aqi"] == 100.0
        assert data["is_stale"] is True
        assert "pollutants" in data
        assert "weather" in data

    def test_bootstrap_matches_csv_observation_parity(self):
        from src.config import PROCESSED_DATA_DIR
        from src.inference.predictor import AQIPredictor
        pred = AQIPredictor()
        csv_obs = pred.get_latest_observation(dataset_path=PROCESSED_DATA_DIR / "features_v2_weather.csv")
        
        resolver = RuntimeAssetResolver(mode="runtime")
        _, bootstrap_meta = resolver.load_bootstrap_feature_vector()
        
        assert csv_obs["current_aqi"] == bootstrap_meta["current_aqi"]
        assert csv_obs["dominant_pollutant"] == bootstrap_meta["dominant_pollutant"]
        assert csv_obs["pollutants"] == bootstrap_meta["pollutants"]
        assert csv_obs["weather"] == bootstrap_meta["weather"]
        assert csv_obs["input_observed_at"] == bootstrap_meta["input_observed_at"].isoformat()

    def test_api_forecast_endpoint_returns_72_horizons(self):
        app = create_app({"TESTING": True})
        client = app.test_client()
        resp = client.get("/api/forecast")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["forecasts"]) == 72
        assert data["forecast_origin"] == "2026-08-31T07:00:00+00:00"
        assert data["is_stale"] is True

    def test_api_explain_endpoint_parity_with_forecast(self):
        app = create_app({"TESTING": True})
        client = app.test_client()
        forecast_resp = client.get("/api/forecast").get_json()
        h24_forecast = forecast_resp["forecasts"][23]["aqi"]

        explain_resp = client.get("/api/explain?horizon=24&top_k=5").get_json()
        assert explain_resp["predicted_aqi"] == pytest.approx(h24_forecast, abs=0.1)
        assert len(explain_resp["top_features"]) == 5
        assert explain_resp["horizon"] == 24


class TestFrontendHardening:
    """Test suite verifying frontend data client error handling and fallback toggling."""

    def test_frontend_client_raises_clean_error_when_api_down_and_fallback_disabled(self):
        client = DashboardDataClient(
            api_base_url="http://127.0.0.1:59999/api",
            timeout=0.1,
            enable_local_fallback=False,
        )
        with patch("requests.get", side_effect=ConnectionError("Refused")):
            with pytest.raises(ConnectionError) as exc_info:
                client.fetch_current()
            assert "starting from standby or unreachable" in str(exc_info.value)
