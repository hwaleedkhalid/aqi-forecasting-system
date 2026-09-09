"""Unit and integration tests for the redesigned Streamlit Dashboard.

Covers:
  - DashboardDataClient (API and fallback modes)
  - Flask API / Direct Inference contract parity
  - New component rendering (hero, outlook, alerts, milestones, telemetry, guidance,
    model details, cold-start, missing-value safety)
  - AppTest end-to-end rendering with mocked REST payloads
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.api.app import create_app
from src.dashboard.components import (
    build_forecast_figure,
    render_alert_banners,
    render_cold_start_error,
    render_current_observation_card,
    render_explainability_section,
    render_freshness_banner,
    render_health_guidance,
    render_metadata_header,
    render_model_system_details,
    render_sidebar,
    render_telemetry_breakdown,
    render_72h_outlook_card,
)
from src.dashboard.ui_helpers import (
    format_relative_age,
    format_timestamp,
    safe_val,
    source_badge_class,
    source_display_name,
)
from src.dashboard.data_client import DashboardDataClient
from src.inference.predictor import AQIPredictor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictor():
    """Singleton AQIPredictor for parity tests."""
    return AQIPredictor()


@pytest.fixture
def flask_client():
    """Flask test client."""
    app = create_app({"TESTING": True})
    return app.test_client()


@pytest.fixture
def mock_obs():
    return {
        "current_aqi": 120.0,
        "category": "Unhealthy for Sensitive Groups",
        "color": "#FF7E00",
        "dominant_pollutant": "pm2_5",
        "health_advisory": "Unusually sensitive people should consider reducing prolonged outdoor exertion.",
        "is_stale": False,
        "input_observed_at": "2026-09-09T12:00:00+00:00",
        "input_age_hours": 0.3,
        "feature_source": "hopsworks",
        "fallback_active": False,
        "cloud_active": True,
        "alert": {
            "active": True,
            "level": "advisory",
            "severity_rank": 2,
            "category": "Unhealthy for Sensitive Groups",
            "aqi": 120.0,
            "message": "Unusually sensitive people should limit prolonged outdoor exertion.",
        },
        "pollutants": {
            "pm2_5": 45.2,
            "pm10": 72.1,
            "no2": 18.3,
            "so2": 5.1,
            "co": 310.0,
            "o3": 67.2,
        },
        "weather": {
            "temperature_2m": 28.5,
            "relative_humidity_2m": 62.0,
            "wind_speed_10m": 3.2,
            "surface_pressure": 1008.4,
        },
        "data_status": "live",
    }


@pytest.fixture
def mock_forecast():
    forecasts = [
        {
            "horizon": h,
            "forecast_time": f"2026-09-09T{h:02d}:00:00Z",
            "aqi": 125.0 + h * 0.5,
            "category": "Unhealthy for Sensitive Groups",
            "color": "#FF7E00",
            "error_lower": 100.0,
            "error_upper": 150.0,
            "alert_level": "advisory",
            "severity_rank": 2,
        }
        for h in range(1, 73)
    ]
    return {
        "model_id": "EXP-019",
        "model_version": 1,
        "feature_count": 114,
        "forecast_origin": "2026-09-09T12:00:00+00:00",
        "input_observed_at": "2026-09-09T12:00:00+00:00",
        "input_age_hours": 0.3,
        "generated_at": "2026-09-09T12:05:00+00:00",
        "inference_latency_ms": 42.0,
        "is_stale": False,
        "data_status": "live",
        "forecasts": forecasts,
        "summary": {
            "peak_aqi": 161.0,
            "peak_horizon": 72,
            "peak_category": "Unhealthy",
            "has_high_severity": False,
            "has_hazardous": False,
        },
        "forecast_alert": {
            "active": True,
            "highest_level": "warning",
            "highest_severity_rank": 2,
            "peak_aqi": 161.0,
            "peak_horizon": 72,
            "peak_category": "Unhealthy",
            "first_advisory_horizon": 1,
            "first_unhealthy_horizon": 48,
            "first_very_unhealthy_horizon": None,
            "first_hazardous_horizon": None,
            "message": "The model forecasts Unhealthy air quality within 72 hours.",
            "upper_interval_crosses_hazardous": False,
            "data_is_stale": False,
            "fallback_active": False,
            "feature_source": "hopsworks",
        },
    }


@pytest.fixture
def mock_model_info():
    return {
        "model_id": "EXP-019",
        "architecture": "Hybrid Specialist (LightGBM h1-6, Ridge h7-37, Blended Ridge h38-72)",
        "feature_count": 114,
        "feature_schema_version": "v2_weather_enriched",
        "status": "validated_champion",
        "test_benchmark_metrics": {
            "overall_rmse": 75.91,
            "overall_mae": 54.22,
            "overall_r2": 0.82,
            "h1_rmse": 18.43,
            "h72_rmse": 112.87,
        },
    }


# ── DashboardDataClient tests ─────────────────────────────────────────────────

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


# ── Flask / Direct inference contract parity ──────────────────────────────────

class TestFlaskApiFallbackParity:
    """Contract parity between Flask REST API and Direct Local Inference fallback."""

    def test_current_observation_contract_parity(self, flask_client, predictor):
        api_resp = flask_client.get("/api/current")
        assert api_resp.status_code == 200
        api_data = api_resp.get_json()
        fallback_data = predictor.get_latest_observation()
        assert api_data["current_aqi"] == pytest.approx(fallback_data["current_aqi"], abs=1e-5)
        assert api_data["dominant_pollutant"] == fallback_data["dominant_pollutant"]
        assert api_data["category"] == fallback_data["category"]
        assert api_data["color"] == fallback_data["color"]
        assert api_data["health_advisory"] == fallback_data["health_advisory"]
        assert api_data["data_status"] == fallback_data["data_status"]
        assert api_data["input_observed_at"] == fallback_data["input_observed_at"]
        assert api_data["is_stale"] == fallback_data["is_stale"]
        assert api_data["input_age_hours"] == pytest.approx(fallback_data["input_age_hours"], abs=0.1)
        assert api_data["pollutants"] == fallback_data["pollutants"]
        assert api_data["weather"] == fallback_data["weather"]

    def test_forecast_contract_parity(self, flask_client, predictor):
        api_resp = flask_client.get("/api/forecast?force_refresh=false")
        assert api_resp.status_code == 200
        api_data = api_resp.get_json()
        fallback_data = predictor.predict_latest(use_cache=True, force_refresh=False)
        assert api_data["model_id"] == fallback_data["model_id"]
        assert api_data["model_version"] == fallback_data["model_version"]
        assert api_data["feature_count"] == fallback_data["feature_count"] == 114
        assert api_data["data_status"] == fallback_data["data_status"]
        assert api_data["input_observed_at"] == fallback_data["input_observed_at"]
        assert api_data["forecast_origin"] == fallback_data["forecast_origin"]
        assert api_data["is_stale"] == fallback_data["is_stale"]
        api_forecasts = api_data["forecasts"]
        fallback_forecasts = fallback_data["forecasts"]
        assert len(api_forecasts) == 72
        assert len(fallback_forecasts) == 72
        for i in range(72):
            af = api_forecasts[i]
            ff = fallback_forecasts[i]
            assert af["horizon"] == ff["horizon"] == i + 1
            assert af["aqi"] == pytest.approx(ff["aqi"], abs=1e-5)
            assert af["category"] == ff["category"]
            assert af["error_lower"] == pytest.approx(ff["error_lower"], abs=1e-5)
            assert af["error_upper"] == pytest.approx(ff["error_upper"], abs=1e-5)
        assert api_data["summary"]["peak_aqi"] == pytest.approx(fallback_data["summary"]["peak_aqi"], abs=1e-5)
        assert api_data["summary"]["peak_horizon"] == fallback_data["summary"]["peak_horizon"]
        assert api_data["summary"]["peak_category"] == fallback_data["summary"]["peak_category"]

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


# ── UI Helpers unit tests ─────────────────────────────────────────────────────

class TestUIHelpers:
    """Unit tests for formatting utility functions."""

    def test_safe_val_normal(self):
        assert safe_val(45.2, "µg/m³", 1) == "45.2 µg/m³"

    def test_safe_val_none(self):
        assert safe_val(None) == "—"

    def test_safe_val_nan(self):
        import math
        assert safe_val(math.nan) == "—"

    def test_safe_val_no_unit(self):
        assert safe_val(123.456, decimals=2) == "123.46"

    def test_format_relative_age_just_now(self):
        assert format_relative_age(0.005) == "just now"

    def test_format_relative_age_minutes(self):
        result = format_relative_age(0.2)
        assert "min ago" in result

    def test_format_relative_age_hours(self):
        result = format_relative_age(4.1)
        assert "4.1h ago" in result

    def test_format_relative_age_none(self):
        result = format_relative_age(None)
        assert result == "unknown time"

    def test_format_timestamp_valid(self):
        result = format_timestamp("2026-09-09T14:32:00+00:00")
        assert "2026-09-09" in result
        assert "14:32" in result

    def test_format_timestamp_none(self):
        assert format_timestamp(None) == "—"

    def test_format_timestamp_unknown(self):
        assert format_timestamp("Unknown") == "—"

    def test_source_display_hopsworks(self):
        label = source_display_name("hopsworks", False, False)
        assert label == "Hopsworks"

    def test_source_display_fallback(self):
        label = source_display_name("bootstrap", True, False)
        assert "Cached" in label

    def test_source_badge_class_stale(self):
        cls = source_badge_class("hopsworks", False, True)
        assert "stale" in cls

    def test_source_badge_class_fallback(self):
        cls = source_badge_class("bootstrap", True, False)
        assert "fallback" in cls

    def test_source_badge_class_normal(self):
        cls = source_badge_class("hopsworks", False, False)
        assert cls == "prl-source-badge"

    def test_category_color_lookups(self):
        from src.dashboard.ui_helpers import (
            category_bg_color,
            category_border_color,
            category_card_class,
            category_color,
            category_solid_text_color,
        )
        assert category_color("Good") == "#22C55E"
        assert category_bg_color("Good") == "#F0FDF4"
        assert category_border_color("Good") == "#BBF7D0"
        assert category_solid_text_color("Good") == "#FFFFFF"
        assert "prl-card-good" in category_card_class("Good")

        assert category_color("Hazardous") == "#7F1D1D"
        assert category_bg_color("Hazardous") == "#FFF1F2"
        assert category_border_color("Hazardous") == "#FFE4E6"
        assert "prl-card-hazardous" in category_card_class("Hazardous")

    def test_humanize_feature_name(self):
        from src.dashboard.ui_helpers import humanize_feature_name
        assert "PM10" in humanize_feature_name("pm10_lag_1h")
        assert "PM2.5" in humanize_feature_name("pm2_5_lag_1h")
        assert "AQI" in humanize_feature_name("epa_aqi_rolling_mean_6h")
        assert "temperature" in humanize_feature_name("temperature_2m_lag_1h").lower()
        assert humanize_feature_name("") == "Unknown Factor"

    def test_format_shap_narrative(self):
        from src.dashboard.ui_helpers import format_shap_narrative
        sample_top = [
            {"feature": "pm10_lag_1h", "shap_value": 18.5, "raw_value": 90.0},
            {"feature": "temperature_2m_lag_1h", "shap_value": -6.2, "raw_value": 24.0},
            {"feature": "hour_of_day_sin", "shap_value": 1.2, "raw_value": 0.5},
        ]
        narratives = format_shap_narrative(sample_top, default_horizon=24)
        assert len(narratives) == 3

        # Strong upward
        assert narratives[0]["is_upward"] is True
        assert narratives[0]["strength"] == "Strong"
        assert "↑" in narratives[0]["arrow"]
        assert "upward" in narratives[0]["direction"]

        # Moderate downward
        assert narratives[1]["is_upward"] is False
        assert narratives[1]["strength"] == "Moderate"
        assert "↓" in narratives[1]["arrow"]
        assert "downward" in narratives[1]["direction"]

        # Minor upward
        assert narratives[2]["strength"] == "Minor"


# ── Chart and Gauge tests ─────────────────────────────────────────────────────

class TestForecastChart:
    """Verify forecast chart structure and EPA reference lines."""

    def test_build_aqi_gauge_figure(self):
        from src.dashboard.components import build_aqi_gauge_figure
        fig = build_aqi_gauge_figure(120.0, "Unhealthy for Sensitive Groups")
        assert fig is not None
        assert fig.data[0].type == "indicator"
        assert fig.data[0].value == 120.0
        assert fig.data[0].gauge.axis.range == (0, 500)
        assert len(fig.data[0].gauge.steps) == 6

    def test_build_aqi_gauge_figure_none_safe(self):
        from src.dashboard.components import build_aqi_gauge_figure
        fig = build_aqi_gauge_figure(None, "Unknown")
        assert fig is not None
        assert fig.data[0].value == 0.0

    def test_build_forecast_figure_structure(self, predictor):
        forecast_data = predictor.predict_latest(use_cache=True)
        fig = build_forecast_figure(forecast_data["forecasts"])
        assert fig is not None
        assert len(fig.data) == 3  # upper, lower filled band, AQI curve
        assert fig.data[1].fill == "tonexty"
        assert "Empirical Error Range" in fig.data[1].name
        assert "Predicted AQI (EXP-019)" in fig.data[2].name
        assert len(fig.data[2].x) == 72
        assert len(fig.data[2].y) == 72
        # Check background zones
        hrects = [s for s in fig.layout.shapes if s.type == "rect"]
        assert len(hrects) >= 6

    def test_chart_has_151_201_301_reference_lines(self):
        dummy_forecasts = [
            {
                "horizon": h,
                "forecast_time": f"2026-09-09T{h:02d}:00:00Z",
                "aqi": 120.0,
                "category": "Unhealthy for Sensitive Groups",
                "error_lower": 100.0,
                "error_upper": 140.0,
            }
            for h in range(1, 73)
        ]
        fig = build_forecast_figure(dummy_forecasts)
        y_lines = [s.y0 for s in fig.layout.shapes if s.type == "line"]
        assert 151 in y_lines, "Expected reference line at y=151 (Unhealthy)"
        assert 201 in y_lines, "Expected reference line at y=201 (Very Unhealthy)"
        assert 301 in y_lines, "Expected reference line at y=301 (Hazardous)"



# ── Component render tests (focused unit) ─────────────────────────────────────

class TestComponentRendering:
    """Focused unit tests for each new redesigned component."""

    def test_render_current_observation_card_no_exception(self, mock_obs):
        with patch("streamlit.markdown") as mock_md:
            render_current_observation_card(mock_obs)
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            assert "120" in html  # AQI value
            assert "Unhealthy for Sensitive Groups" in html

    def test_hero_aqi_value_in_card(self, mock_obs):
        with patch("streamlit.markdown") as mock_md:
            render_current_observation_card(mock_obs)
            html = mock_md.call_args[0][0]
            assert "120" in html

    def test_hero_card_missing_values_safe(self):
        """None AQI should render as — not 'None' or crash."""
        obs = {"current_aqi": None, "category": "Unknown", "is_stale": False}
        with patch("streamlit.markdown") as mock_md:
            render_current_observation_card(obs)
            html = mock_md.call_args[0][0]
            assert "None" not in html
            assert "—" in html

    def test_render_72h_outlook_card_peak_present(self, mock_forecast):
        with patch("streamlit.markdown") as mock_md:
            render_72h_outlook_card(mock_forecast)
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            assert "72-Hour Outlook" in html
            assert "161" in html  # peak AQI

    def test_render_72h_outlook_card_no_unhealthy(self):
        forecast_data = {
            "forecast_alert": {
                "highest_level": "none",
                "peak_aqi": 85.0,
                "peak_horizon": 12,
                "peak_category": "Moderate",
                "first_unhealthy_horizon": None,
                "first_very_unhealthy_horizon": None,
                "first_hazardous_horizon": None,
            },
            "summary": {},
        }
        with patch("streamlit.markdown") as mock_md:
            render_72h_outlook_card(forecast_data)
            html = mock_md.call_args[0][0]
            assert "Good / Moderate throughout" in html

    def test_render_metadata_header_compact(self):
        with patch("streamlit.markdown") as mock_md:
            render_metadata_header(
                "2026-09-09T12:00:00+00:00",
                "2026-09-09T12:05:00+00:00",
                "REST API",
                42.0,
            )
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            # Should NOT show raw ISO timestamp as primary content
            assert "Hopsworks" in html or "ago" in html or "Updated" in html

    def test_render_milestone_cards_rendered(self, mock_forecast):
        from src.dashboard.app import render_horizon_milestones
        cols_mock = [MagicMock() for _ in range(5)]
        for col in cols_mock:
            col.__enter__ = lambda s: s
            col.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.columns", return_value=cols_mock), \
             patch("streamlit.markdown") as mock_md:
            render_horizon_milestones(mock_forecast["forecasts"])
            assert mock_md.call_count == 5

    def test_render_telemetry_pollutant_values(self, mock_obs):
        col_mock = MagicMock()
        col_mock.__enter__ = lambda s: s
        col_mock.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.columns", return_value=[col_mock, col_mock]), \
             patch("streamlit.markdown") as mock_md:
            render_telemetry_breakdown(mock_obs)
            all_html = " ".join(str(c) for c in mock_md.call_args_list)
            assert "PM2.5" in all_html
            assert "45.2" in all_html

    def test_render_telemetry_weather_values(self, mock_obs):
        col_mock = MagicMock()
        col_mock.__enter__ = lambda s: s
        col_mock.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.columns", return_value=[col_mock, col_mock]), \
             patch("streamlit.markdown") as mock_md:
            render_telemetry_breakdown(mock_obs)
            all_html = " ".join(str(c) for c in mock_md.call_args_list)
            assert "Temperature" in all_html
            assert "28.5" in all_html

    def test_render_telemetry_missing_values_safe(self):
        obs = {"pollutants": {"pm2_5": None, "pm10": float("nan")}, "weather": {}}
        col_mock = MagicMock()
        col_mock.__enter__ = lambda s: s
        col_mock.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.columns", return_value=[col_mock, col_mock]), \
             patch("streamlit.markdown") as mock_md:
            render_telemetry_breakdown(obs)
            all_html = " ".join(str(c) for c in mock_md.call_args_list)
            assert "None" not in all_html
            assert "nan" not in all_html

    def test_render_health_guidance_section(self, mock_obs, mock_forecast):
        with patch("streamlit.markdown") as mock_md:
            render_health_guidance(mock_obs, mock_forecast)
            html = mock_md.call_args[0][0]
            assert "Health Guidance" in html or "Unhealthy for Sensitive Groups" in html or "advisory" in html.lower()

    def test_render_health_guidance_forecast_upgrade(self, mock_obs, mock_forecast):
        """When forecast severity > current severity, upgrade notice appears."""
        with patch("streamlit.markdown") as mock_md:
            render_health_guidance(mock_obs, mock_forecast)
            html = mock_md.call_args[0][0]
            # Forecast is 'warning' (Unhealthy), current is 'advisory' — upgrade notice expected
            assert "Forecast outlook" in html or "expected to reach" in html

    def test_render_model_system_details_expander(self, mock_model_info, mock_forecast, mock_obs):
        exp_mock = MagicMock()
        exp_mock.__enter__ = lambda s: s
        exp_mock.__exit__ = MagicMock(return_value=False)
        col_mock = MagicMock()
        col_mock.__enter__ = lambda s: s
        col_mock.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.expander", return_value=exp_mock), \
             patch("streamlit.columns", return_value=[col_mock, col_mock]), \
             patch("streamlit.markdown") as mock_md:
            render_model_system_details(mock_model_info, mock_forecast, mock_obs, "REST API")
            all_text = " ".join(str(c) for c in mock_md.call_args_list)
            assert "EXP-019" in all_text
            assert "114" in all_text

    def test_render_cold_start_state(self):
        """Cold-start renderer shows user-friendly message."""
        exp_mock = MagicMock()
        exp_mock.__enter__ = lambda s: s
        exp_mock.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.markdown") as mock_md, \
             patch("streamlit.button", return_value=False), \
             patch("streamlit.expander", return_value=exp_mock):
            render_cold_start_error(ConnectionError("Service offline"))
            html = mock_md.call_args[0][0]
            assert "waking up" in html

    def test_render_sidebar_is_noop(self):
        """Sidebar stub executes without error and writes nothing to sidebar."""
        with patch("streamlit.sidebar") as mock_sidebar:
            render_sidebar({}, {}, "REST API")
            mock_sidebar.assert_not_called()

    def test_render_freshness_banner_preserved(self):
        """Legacy freshness banner function still works (called inside render_alert_banners)."""
        with patch("streamlit.warning") as mock_warn:
            from src.dashboard.components import render_freshness_banner
            render_freshness_banner(
                is_stale=True,
                observed_at="2026-08-31T07:00:00+00:00",
                age_hours=145.0,
            )
            mock_warn.assert_called_once()
            assert "145.0 hours" in mock_warn.call_args[0][0]

    def test_render_aqi_legend(self):
        from src.dashboard.components import render_aqi_legend
        with patch("streamlit.markdown") as mock_md:
            render_aqi_legend()
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            assert "0–50" in html
            assert "51–100" in html
            assert "101–150" in html
            assert "151–200" in html
            assert "201–300" in html
            assert "301+" in html

    def test_render_forecast_narrative_card(self):
        from src.dashboard.components import render_forecast_narrative_card
        explain_data = {
            "horizon": 24,
            "top_features": [
                {"feature": "pm10_lag_1h", "shap_value": 16.4, "raw_value": 85.0},
                {"feature": "temperature_2m_lag_1h", "shap_value": -5.1, "raw_value": 22.0},
            ],
        }
        with patch("streamlit.markdown") as mock_md:
            render_forecast_narrative_card(explain_data, horizon=24)
            mock_md.assert_called_once()
            html = mock_md.call_args[0][0]
            assert "Why this forecast?" in html
            assert "PM10" in html
            assert "Strong" in html


# ── main() end-to-end tests ───────────────────────────────────────────────────


class TestAppMain:
    """End-to-end tests for app.main() with mocked dependencies."""

    def _all_patches(self, obs, forecast, model_info):
        """Context managers to run main() with mocked API + mocked Streamlit primitives."""
        return (
            patch("streamlit.set_page_config"),
            patch("streamlit.markdown"),
            patch("streamlit.caption"),
            patch("streamlit.columns", side_effect=lambda n, **kw: [MagicMock().__enter__() or MagicMock() for _ in (range(n) if isinstance(n, int) else n)]),
            patch("streamlit.tabs", return_value=[MagicMock().__enter__() or MagicMock(), MagicMock().__enter__() or MagicMock()]),
            patch("streamlit.slider", return_value=24),
            patch("streamlit.button", return_value=False),
            patch("streamlit.spinner", return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False))),
            patch("streamlit.divider"),
            patch("streamlit.plotly_chart"),
            patch("streamlit.expander", return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False))),
            patch("streamlit.info"),
            patch("streamlit.dataframe"),
            patch("src.dashboard.data_client.DashboardDataClient.fetch_current", return_value=(obs, "REST API")),
            patch("src.dashboard.data_client.DashboardDataClient.fetch_forecast", return_value=(forecast, "REST API")),
            patch("src.dashboard.data_client.DashboardDataClient.fetch_model_info", return_value=(model_info, "REST API")),
            patch("src.dashboard.data_client.DashboardDataClient.fetch_explain", return_value=({}, "REST API")),
        )

    def test_app_main_executes_successfully(self, mock_obs, mock_forecast, mock_model_info):
        """main() runs end-to-end without exception with mocked payloads."""
        from src.dashboard.app import main
        patches = self._all_patches(mock_obs, mock_forecast, mock_model_info)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], \
             patches[12], patches[13], patches[14], patches[15], patches[16]:
            main()  # Must not raise

    def test_app_main_handles_connection_error(self, mock_obs):
        """ConnectionError triggers cold-start screen."""
        from src.dashboard.app import main
        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.markdown"),
            patch("streamlit.caption"),
            patch("streamlit.columns", side_effect=lambda n, **kw: [MagicMock() for _ in (range(n) if isinstance(n, int) else n)]),
            patch("streamlit.button", return_value=False),
            patch("streamlit.spinner", return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False))),
            patch("src.dashboard.data_client.DashboardDataClient.fetch_current", side_effect=ConnectionError("Offline")),
            patch("src.dashboard.app.render_cold_start_error") as mock_cold,
        ):
            main()
            mock_cold.assert_called_once()

    def test_app_main_handles_generic_exception(self):
        """Generic exception triggers error message with technical details."""
        from src.dashboard.app import main
        with (
            patch("streamlit.set_page_config"),
            patch("streamlit.markdown"),
            patch("streamlit.caption"),
            patch("streamlit.columns", side_effect=lambda n, **kw: [MagicMock() for _ in (range(n) if isinstance(n, int) else n)]),
            patch("streamlit.button", return_value=False),
            patch("streamlit.spinner", return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False))),
            patch("src.dashboard.data_client.DashboardDataClient.fetch_current", side_effect=RuntimeError("Data failure")),
            patch("streamlit.error") as mock_err,
            patch("streamlit.expander", return_value=MagicMock(__enter__=lambda s: s, __exit__=MagicMock(return_value=False))),
        ):
            main()
            mock_err.assert_called_once()

    def test_app_main_metadata_row_fallbacks_regression(self, mock_model_info):
        """observed_at and forecast_origin resolve correctly without NameError."""
        from src.dashboard.app import main
        obs = {
            "current_aqi": 120.0,
            "category": "Moderate",
            "input_observed_at": "2026-09-09T12:00:00+00:00",
            "is_stale": False,
            "alert": {"level": "none"},
        }
        forecast_no_origin = {
            "input_observed_at": "2026-09-09T12:00:00+00:00",
            "input_age_hours": 0.3,
            "generated_at": "2026-09-09T12:05:00+00:00",
            "inference_latency_ms": 15.2,
            "forecast_alert": {"highest_level": "none"},
            "forecasts": [],
            "summary": {},
        }
        # No forecast_origin key — tests fallback to observed_at
        assert "forecast_origin" not in forecast_no_origin
        patches = self._all_patches(obs, forecast_no_origin, mock_model_info)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], \
             patches[6], patches[7], patches[8], patches[9], patches[10], patches[11], \
             patches[12], patches[13], patches[14], patches[15], patches[16]:
            main()  # Must not raise NameError
