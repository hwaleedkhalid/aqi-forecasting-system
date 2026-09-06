"""Unit and parity tests for Streamlit Dashboard components and client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.api.app import create_app
from src.dashboard.components import (
    build_forecast_figure,
    render_current_observation_card,
    render_freshness_banner,
    render_metadata_header,
    render_sidebar,
    render_telemetry_breakdown,
)
from src.dashboard.app import render_horizon_milestones
from src.dashboard.data_client import DashboardDataClient
from src.inference.predictor import AQIPredictor


@pytest.fixture
def predictor():
    """Create a singleton AQIPredictor instance for parity tests."""
    return AQIPredictor()


@pytest.fixture
def flask_client():
    """Create Flask test client."""
    app = create_app({"TESTING": True})
    return app.test_client()


class TestDashboardDataClient:
    """Test suite for DashboardDataClient in both API and fallback modes."""

    def test_is_api_online_true(self):
        client = DashboardDataClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"service_ready": True}

        with patch("requests.get", return_value=mock_resp):
            assert client.is_api_online() is True

    def test_is_api_online_false_on_http_error(self):
        client = DashboardDataClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("requests.get", return_value=mock_resp):
            assert client.is_api_online() is False

    def test_is_api_online_false_on_exception(self):
        client = DashboardDataClient()
        with patch("requests.get", side_effect=requests.ConnectionError("Connection refused")):
            assert client.is_api_online() is False

    def test_fetch_current_api_mode(self):
        client = DashboardDataClient()
        mock_payload = {"current_aqi": 185.0, "data_status": "stale"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload

        with patch("requests.get", return_value=mock_resp):
            data, source = client.fetch_current()
            assert data["current_aqi"] == 185.0
            assert source == "REST API"

    def test_fetch_current_fallback_mode(self, predictor):
        client = DashboardDataClient(predictor=predictor)
        with patch("requests.get", side_effect=requests.ConnectionError("Offline")):
            data, source = client.fetch_current()
            assert "current_aqi" in data
            assert "Direct Local Inference" in source

    def test_fetch_forecast_api_mode(self):
        client = DashboardDataClient()
        mock_payload = {"model_id": "EXP-019", "forecasts": [{"horizon": 1, "aqi": 190.0}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload

        with patch("requests.get", return_value=mock_resp) as mock_get:
            data, source = client.fetch_forecast(force_refresh=True)
            assert data["model_id"] == "EXP-019"
            assert source == "REST API"
            mock_get.assert_called_once()
            _, kwargs = mock_get.call_args
            assert kwargs["params"] == {"force_refresh": "true"}

    def test_fetch_forecast_fallback_mode(self, predictor):
        client = DashboardDataClient(predictor=predictor)
        with patch("requests.get", side_effect=requests.ConnectionError("Offline")):
            data, source = client.fetch_forecast(force_refresh=False)
            assert data["model_id"] == "EXP-019"
            assert len(data["forecasts"]) == 72
            assert "Direct Local Inference" in source

    def test_fetch_model_info_api_mode(self):
        client = DashboardDataClient()
        mock_payload = {"model_id": "EXP-019", "feature_count": 114}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_payload

        with patch("requests.get", return_value=mock_resp):
            data, source = client.fetch_model_info()
            assert data["feature_count"] == 114
            assert source == "REST API"

    def test_fetch_model_info_fallback_mode(self, predictor):
        client = DashboardDataClient(predictor=predictor)
        with patch("requests.get", side_effect=requests.ConnectionError("Offline")):
            data, source = client.fetch_model_info()
            assert data["feature_count"] == 114
            assert "Direct Local Inference" in source


class TestFlaskApiFallbackParity:
    """Rigorous contract parity testing between Flask REST API and Direct Local Inference fallback."""

    def test_current_observation_contract_parity(self, flask_client, predictor):
        # 1. Fetch via REST API
        api_resp = flask_client.get("/api/current")
        assert api_resp.status_code == 200
        api_data = api_resp.get_json()

        # 2. Fetch via Direct Local Fallback
        fallback_data = predictor.get_latest_observation()

        # Verify key parity
        assert api_data["current_aqi"] == pytest.approx(fallback_data["current_aqi"], abs=1e-5)
        assert api_data["dominant_pollutant"] == fallback_data["dominant_pollutant"]
        assert api_data["category"] == fallback_data["category"]
        assert api_data["color"] == fallback_data["color"]
        assert api_data["health_advisory"] == fallback_data["health_advisory"]
        assert api_data["data_status"] == fallback_data["data_status"]
        assert api_data["input_observed_at"] == fallback_data["input_observed_at"]
        assert api_data["is_stale"] == fallback_data["is_stale"]
        assert api_data["input_age_hours"] == pytest.approx(fallback_data["input_age_hours"], abs=0.1)

        # Verify telemetry measurements parity
        assert api_data["pollutants"] == fallback_data["pollutants"]
        assert api_data["weather"] == fallback_data["weather"]

    def test_forecast_contract_parity(self, flask_client, predictor):
        # 1. Fetch via REST API
        api_resp = flask_client.get("/api/forecast?force_refresh=false")
        assert api_resp.status_code == 200
        api_data = api_resp.get_json()

        # 2. Fetch via Direct Local Fallback
        fallback_data = predictor.predict_latest(use_cache=True, force_refresh=False)

        # Verify top-level metadata parity
        assert api_data["model_id"] == fallback_data["model_id"]
        assert api_data["model_version"] == fallback_data["model_version"]
        assert api_data["feature_count"] == fallback_data["feature_count"] == 114
        assert api_data["data_status"] == fallback_data["data_status"]
        assert api_data["input_observed_at"] == fallback_data["input_observed_at"]
        assert api_data["forecast_origin"] == fallback_data["forecast_origin"]
        assert api_data["is_stale"] == fallback_data["is_stale"]

        # Verify 72-hour forecast items parity
        api_forecasts = api_data["forecasts"]
        fallback_forecasts = fallback_data["forecasts"]
        assert len(api_forecasts) == 72
        assert len(fallback_forecasts) == 72

        for i in range(72):
            af = api_forecasts[i]
            ff = fallback_forecasts[i]
            assert af["horizon"] == ff["horizon"] == i + 1
            assert af["forecast_time"] == ff["forecast_time"]
            assert af["aqi"] == pytest.approx(ff["aqi"], abs=1e-5)
            assert af["category"] == ff["category"]
            assert af["color"] == ff["color"]
            assert af["error_lower"] == pytest.approx(ff["error_lower"], abs=1e-5)
            assert af["error_upper"] == pytest.approx(ff["error_upper"], abs=1e-5)

        # Verify summary parity
        assert api_data["summary"]["peak_aqi"] == pytest.approx(fallback_data["summary"]["peak_aqi"], abs=1e-5)
        assert api_data["summary"]["peak_horizon"] == fallback_data["summary"]["peak_horizon"]
        assert api_data["summary"]["peak_category"] == fallback_data["summary"]["peak_category"]
        assert api_data["summary"]["has_high_severity"] == fallback_data["summary"]["has_high_severity"]
        assert api_data["summary"]["has_hazardous"] == fallback_data["summary"]["has_hazardous"]

    def test_model_info_contract_parity(self, flask_client, predictor):
        api_resp = flask_client.get("/api/model/info")
        assert api_resp.status_code == 200
        api_data = api_resp.get_json()

        fallback_data = predictor.get_model_metadata()

        assert api_data["model_id"] == fallback_data["model_id"] == "EXP-019"
        assert api_data["architecture"] == fallback_data["architecture"]
        assert api_data["feature_count"] == fallback_data["feature_count"] == 114
        assert api_data["status"] == fallback_data["status"]
        assert api_data["test_benchmark_metrics"] == fallback_data["test_benchmark_metrics"]


class TestDashboardComponents:
    """Test suite for UI components and Plotly chart construction."""

    def test_build_forecast_figure_structure(self, predictor):
        forecast_data = predictor.predict_latest(use_cache=True)
        fig = build_forecast_figure(forecast_data["forecasts"])

        # Check figure properties
        assert fig is not None
        assert len(fig.data) == 3  # upper line, lower filled band, predicted AQI curve

        # Trace 0: upper bound (invisible)
        assert fig.data[0].mode == "lines"
        assert fig.data[0].showlegend is False

        # Trace 1: shaded empirical error band
        assert fig.data[1].fill == "tonexty"
        assert "Empirical Error Range" in fig.data[1].name

        # Trace 2: Predicted AQI
        assert "Predicted AQI (EXP-019)" in fig.data[2].name
        assert len(fig.data[2].x) == 72
        assert len(fig.data[2].y) == 72

        # Verify horizontal reference lines at 200 and 300
        shapes = fig.layout.shapes
        y_lines = [s.y0 for s in shapes if s.type == "line"]
        assert 200 in y_lines
        assert 300 in y_lines

    def test_components_execute_without_exception(self, predictor):
        """Verify components can be called without runtime errors."""
        obs = predictor.get_latest_observation()
        forecast_data = predictor.predict_latest(use_cache=True)
        model_info = predictor.get_model_metadata()

        with patch("streamlit.warning") as mock_warn:
            render_freshness_banner(is_stale=True, observed_at="2026-08-31T07:00:00+00:00", age_hours=145.0)
            mock_warn.assert_called_once()

        with patch("streamlit.columns", return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()]):
            render_metadata_header("2026-08-31T07:00:00+00:00", "2026-09-06T14:00:00+00:00", "REST API", 12.5)

        with patch("streamlit.markdown") as mock_md:
            render_current_observation_card(obs)
            mock_md.assert_called_once()

        with patch("streamlit.columns", side_effect=[[MagicMock()]*6, [MagicMock()]*4]):
            render_telemetry_breakdown(obs)

        with patch("streamlit.sidebar"):
            render_sidebar(model_info, forecast_data["summary"], "REST API")

        with patch("streamlit.columns", return_value=[MagicMock()]*5):
            render_horizon_milestones(forecast_data["forecasts"])

    def test_app_main_executes_successfully(self):
        """Verify main() runs end-to-end with mocked Streamlit presentation layer."""
        from src.dashboard.app import main

        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.caption"),
            patch("streamlit.columns", side_effect=lambda n: [MagicMock()]*n if isinstance(n, int) else [MagicMock()]*len(n)),
            patch("streamlit.slider", return_value=24),
            patch("streamlit.button", return_value=False),
            patch("streamlit.spinner"),
            patch("streamlit.markdown"),
            patch("streamlit.subheader"),
            patch("streamlit.plotly_chart"),
            patch("src.dashboard.app.render_freshness_banner"),
            patch("src.dashboard.app.render_metadata_header"),
            patch("src.dashboard.app.render_current_observation_card"),
            patch("src.dashboard.app.render_horizon_milestones"),
            patch("src.dashboard.app.render_telemetry_breakdown"),
            patch("src.dashboard.app.render_explainability_section"),
            patch("src.dashboard.app.render_sidebar"),
        ):
            main()

    def test_app_main_handles_exception(self):
        """Verify main() gracefully displays error on client failure."""
        from src.dashboard.app import main

        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.title"),
            patch("streamlit.caption"),
            patch("streamlit.columns", side_effect=lambda n: [MagicMock()]*n if isinstance(n, int) else [MagicMock()]*len(n)),
            patch("streamlit.button", return_value=False),
            patch("src.dashboard.data_client.DashboardDataClient.fetch_current", side_effect=RuntimeError("Data failure")),
            patch("streamlit.error") as mock_err,
        ):
            main()
            mock_err.assert_called_once()

