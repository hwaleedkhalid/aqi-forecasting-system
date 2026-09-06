"""Unit and integration tests for src.api (Flask REST API)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from unittest.mock import patch
import pytest

from src.api.app import create_app
from src.api.routes import parse_bool_arg
from werkzeug.exceptions import BadRequest


@pytest.fixture
def app():
    """Create test application instance."""
    app = create_app({
        "TESTING": True,
        "CORS_ORIGINS": ["http://localhost:8501", "http://127.0.0.1:8501"],
    })
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()


class TestParseBoolArg:
    """Test suite for query string boolean parser."""

    def test_parses_valid_true_variants(self):
        assert parse_bool_arg("true") is True
        assert parse_bool_arg("TRUE") is True
        assert parse_bool_arg("1") is True

    def test_parses_valid_false_variants(self):
        assert parse_bool_arg("false") is False
        assert parse_bool_arg("FALSE") is False
        assert parse_bool_arg("0") is False

    def test_parses_none_returns_default(self):
        assert parse_bool_arg(None, default=False) is False
        assert parse_bool_arg(None, default=True) is True

    def test_invalid_string_raises_bad_request(self):
        with pytest.raises(BadRequest, match="Invalid boolean"):
            parse_bool_arg("maybe")
        with pytest.raises(BadRequest, match="Invalid boolean"):
            parse_bool_arg("yes")
        with pytest.raises(BadRequest, match="Invalid boolean"):
            parse_bool_arg("2")


class TestAPIEndpoints:
    """Test suite for Flask API endpoints."""

    def test_health_healthy(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["service_ready"] is True
        assert data["model_loaded"] is True
        assert data["model_id"] == "EXP-019"
        assert "timestamp" in data

    def test_health_degraded_when_model_missing(self, client):
        with patch("src.inference.model_loader.ModelLoader.load_model", side_effect=FileNotFoundError("Model missing")):
            resp = client.get("/api/health")
            assert resp.status_code == 503
            data = resp.get_json()
            assert data["status"] == "degraded"
            assert data["service_ready"] is False
            assert data["model_loaded"] is False

    def test_current_endpoint_does_not_require_forecast_generation(self, client):
        resp = client.get("/api/current")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "current_aqi" in data
        assert "dominant_pollutant" in data
        assert "category" in data
        assert "color" in data
        assert "pollutants" in data
        assert "weather" in data
        assert data["data_status"] in ["live", "stale"]
        assert "input_observed_at" in data
        assert "retrieved_at" in data
        # Verify forecast keys are NOT present
        assert "forecasts" not in data

    def test_forecast_endpoint_returns_200_and_correct_contract(self, client):
        resp = client.get("/api/forecast")
        assert resp.status_code == 200
        data = resp.get_json()

        # Check root provenance and freshness keys
        assert data["model_id"] == "EXP-019"
        assert data["model_version"] == "1.0-production"
        assert data["feature_count"] == 114
        assert data["data_status"] in ["live", "stale"]
        assert "input_observed_at" in data
        assert "forecast_origin" in data
        assert "generated_at" in data
        assert "input_age_hours" in data
        assert "is_stale" in data
        assert "inference_latency_ms" in data

        # Check 72 forecast horizons
        forecasts = data["forecasts"]
        assert len(forecasts) == 72
        assert forecasts[0]["horizon"] == 1
        assert forecasts[71]["horizon"] == 72

        # Check summary
        assert "summary" in data
        assert "peak_aqi" in data["summary"]

    def test_forecast_origin_equals_input_observation_time(self, client):
        resp = client.get("/api/forecast")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["forecast_origin"] == data["input_observed_at"]

    def test_h1_time_is_origin_plus_one_hour(self, client):
        resp = client.get("/api/forecast")
        data = resp.get_json()
        f1 = data["forecasts"][0]
        assert f1["forecast_time"] == "2026-08-31T08:00:00+00:00"

    def test_h72_time_is_origin_plus_72_hours(self, client):
        resp = client.get("/api/forecast")
        data = resp.get_json()
        f72 = data["forecasts"][71]
        assert f72["forecast_time"] == "2026-09-03T07:00:00+00:00"

    def test_generated_at_does_not_shift_forecast_horizons(self, client):
        resp = client.get("/api/forecast")
        data = resp.get_json()
        # generated_at reflects request time
        gen_time = datetime.fromisoformat(data["generated_at"])
        assert gen_time.year == 2026
        # But forecast_time remains anchored to observation time in August
        assert data["forecasts"][0]["forecast_time"].startswith("2026-08-31T08:00:00")

    def test_force_refresh_true_and_false(self, client):
        resp_false = client.get("/api/forecast?force_refresh=false")
        assert resp_false.status_code == 200
        resp_true = client.get("/api/forecast?force_refresh=true")
        assert resp_true.status_code == 200
        # Recomputed forecast from stored features must still report stale
        assert resp_true.get_json()["data_status"] == "stale"

    def test_invalid_boolean_query_returns_400(self, client):
        resp = client.get("/api/forecast?force_refresh=maybe")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "Invalid boolean" in data["error"]
        assert data["status"] == 400

    def test_model_info_returns_canonical_114_features(self, client):
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model_id"] == "EXP-019"
        assert data["feature_count"] == 114
        assert data["status"] == "validated_production_champion"
        assert data["test_benchmark_metrics"]["overall_rmse"] == 75.91

    def test_cors_headers_present(self, client):
        resp = client.get("/api/health", headers={"Origin": "http://localhost:8501"})
        assert resp.status_code == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:8501"

    def test_unknown_endpoint_returns_404_json(self, client):
        resp = client.get("/api/unknown_route")
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["status"] == 404
        assert "Endpoint not found" in data["error"]

    def test_internal_error_does_not_expose_traceback(self, app, client):
        """HTTP 500 must return generic message without stack traces or path leakage."""
        @app.route("/api/test_crash")
        def crash_route():
            raise RuntimeError("Confidential database password or path: D:/secret/data.csv")

        resp = client.get("/api/test_crash")
        assert resp.status_code == 500
        data = resp.get_json()
        assert data["status"] == 500
        assert data["error"] == "Internal server error"
        # Verify no traceback or internal error strings leaked
        assert "Confidential" not in str(data)
        assert "Traceback" not in str(data)
        assert "secret" not in str(data)
