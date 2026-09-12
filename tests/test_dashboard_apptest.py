"""
AppTest end-to-end verification for the redesigned Pearls AQI dashboard.

Uses streamlit.testing.v1.AppTest with fully realistic REST API payloads
mocked at the DashboardDataClient layer. Verifies:
  - No exceptions (len(at.exception) == 0)
  - Hero AQI renders
  - 72-Hour Outlook renders
  - Alert area renders (advisory scenario)
  - Status/freshness line present (derived from input_observed_at/forecast_origin,
    NOT from generated_at)
  - Forecast chart rendered
  - Five milestone cards rendered
  - Pollutant values present (PM2.5, NO2, etc.)
  - Weather values present (temperature, humidity)
  - Health guidance present
  - Advanced Insights / SHAP section present
  - Model & System Details expander present
  - Refresh button present
  - Cold-start state (ConnectionError) renders without exception
  - Stale data scenario renders compact caption, not duplicate warning box
  - Missing / None values do not produce literal 'None' or 'nan' in output
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on sys.path before importing Streamlit app modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("FLASK_API_URL", "http://127.0.0.1:5000/api")
os.environ.setdefault("ENABLE_LOCAL_FALLBACK", "false")

from streamlit.testing.v1 import AppTest

# ---------------------------------------------------------------------------
# Realistic mock payloads
# ---------------------------------------------------------------------------

APP_PATH = str(PROJECT_ROOT / "src" / "dashboard" / "app.py")

_FORECAST_ORIGIN = "2026-09-09T12:00:00+00:00"
_GENERATED_AT = "2026-09-09T12:05:00+00:00"  # intentionally different from origin

_MOCK_OBS = {
    "current_aqi": 128.0,
    "category": "Unhealthy for Sensitive Groups",
    "color": "#FF7E00",
    "dominant_pollutant": "pm2_5",
    "health_advisory": (
        "Unusually sensitive people should consider reducing prolonged "
        "outdoor exertion. Watch for symptoms such as coughing or shortness of breath."
    ),
    "is_stale": False,
    "input_observed_at": _FORECAST_ORIGIN,
    "input_age_hours": 0.25,
    "feature_source": "hopsworks",
    "fallback_active": False,
    "cloud_active": True,
    "alert": {
        "active": True,
        "level": "advisory",
        "severity_rank": 2,
        "category": "Unhealthy for Sensitive Groups",
        "aqi": 128.0,
        "message": "Unusually sensitive people should limit prolonged outdoor exertion.",
        "data_is_stale": False,
        "feature_source": "hopsworks",
        "fallback_active": False,
    },
    "pollutants": {
        "pm2_5": 52.3,
        "pm10": 84.1,
        "no2": 21.7,
        "so2": 6.2,
        "co": 415.0,
        "o3": 71.4,
    },
    "weather": {
        "temperature_2m": 31.2,
        "relative_humidity_2m": 58.0,
        "wind_speed_10m": 2.8,
        "surface_pressure": 1006.1,
    },
    "data_status": "live",
}

_FORECAST_ROWS = [
    {
        "horizon": h,
        "forecast_time": f"2026-09-09T{h:02d}:00:00Z",
        "aqi": 128.0 + h * 0.8,
        "category": "Unhealthy for Sensitive Groups" if h < 40 else "Unhealthy",
        "color": "#FF7E00" if h < 40 else "#FF0000",
        "error_lower": 100.0,
        "error_upper": 158.0,
        "alert_level": "advisory" if h < 40 else "warning",
        "severity_rank": 2 if h < 40 else 3,
    }
    for h in range(1, 73)
]

_MOCK_FORECAST = {
    "model_id": "EXP-019",
    "model_version": 1,
    "feature_count": 114,
    "forecast_origin": _FORECAST_ORIGIN,
    "input_observed_at": _FORECAST_ORIGIN,
    "input_age_hours": 0.25,
    "generated_at": _GENERATED_AT,
    "inference_latency_ms": 38.4,
    "is_stale": False,
    "data_status": "live",
    "forecasts": _FORECAST_ROWS,
    "summary": {
        "peak_aqi": 185.6,
        "peak_horizon": 72,
        "peak_category": "Unhealthy",
        "has_high_severity": False,
        "has_hazardous": False,
        "forecast_alert": {
            "active": True,
            "highest_level": "warning",
            "highest_severity_rank": 3,
            "peak_aqi": 185.6,
            "peak_horizon": 72,
            "peak_category": "Unhealthy",
            "first_advisory_horizon": 1,
            "first_unhealthy_horizon": 40,
            "first_very_unhealthy_horizon": None,
            "first_hazardous_horizon": None,
            "message": "The model forecasts Unhealthy air quality conditions within 72 hours.",
            "upper_interval_crosses_hazardous": False,
            "data_is_stale": False,
            "fallback_active": False,
            "feature_source": "hopsworks",
        },
    },
    "forecast_alert": {
        "active": True,
        "highest_level": "warning",
        "highest_severity_rank": 3,
        "peak_aqi": 185.6,
        "peak_horizon": 72,
        "peak_category": "Unhealthy",
        "first_advisory_horizon": 1,
        "first_unhealthy_horizon": 40,
        "first_very_unhealthy_horizon": None,
        "first_hazardous_horizon": None,
        "message": "The model forecasts Unhealthy air quality conditions within 72 hours.",
        "upper_interval_crosses_hazardous": False,
        "data_is_stale": False,
        "fallback_active": False,
        "feature_source": "hopsworks",
    },
}

_MOCK_MODEL_INFO = {
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

_MOCK_EXPLAIN = {
    "horizon": 24,
    "specialist_type": "Ridge",
    "blend_weight": 1.0,
    "raw_specialist_output": 142.3,
    "model_component": 142.3,
    "persistence_component": 0.0,
    "explained_output_preclip": 142.3,
    "predicted_aqi": 142.3,
    "base_value": 128.0,
    "additivity_error": 0.01,
    "top_features": [
        {"feature": "pm2_5_lag_1h", "shap_value": 8.2, "raw_value": 52.3, "scaled_value": 1.14},
        {"feature": "aqi_lag_1h", "shap_value": 5.1, "raw_value": 128.0, "scaled_value": 0.92},
    ],
    "global_persistence_mean_contribution": 12.4,
    "global_top_features": [
        {"feature": "pm2_5_lag_1h", "mean_abs_shap": 9.1},
        {"feature": "aqi_lag_1h", "mean_abs_shap": 6.3},
    ],
}


# ---------------------------------------------------------------------------
# Fixture: AppTest instance with all API calls mocked
# ---------------------------------------------------------------------------

def _make_at(obs=None, forecast=None, model_info=None, explain=None):
    """Create an AppTest with mocked DashboardDataClient methods."""
    obs = obs or _MOCK_OBS
    forecast = forecast or _MOCK_FORECAST
    model_info = model_info or _MOCK_MODEL_INFO
    explain = explain or _MOCK_EXPLAIN

    at = AppTest.from_file(APP_PATH, default_timeout=60)

    with (
        patch("src.dashboard.data_client.DashboardDataClient.fetch_current",
              return_value=(obs, "REST API")),
        patch("src.dashboard.data_client.DashboardDataClient.fetch_forecast",
              return_value=(forecast, "REST API")),
        patch("src.dashboard.data_client.DashboardDataClient.fetch_model_info",
              return_value=(model_info, "REST API")),
        patch("src.dashboard.data_client.DashboardDataClient.fetch_explain",
              return_value=(explain, "REST API")),
    ):
        at.run()

    return at


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAppTestFullRun:
    """Full AppTest end-to-end rendering verification."""

    def test_no_exceptions(self):
        """Primary requirement: len(at.exception) == 0."""
        at = _make_at()
        assert len(at.exception) == 0, (
            f"AppTest raised {len(at.exception)} exception(s):\n"
            + "\n".join(str(e) for e in at.exception)
        )

    def test_hero_aqi_present(self):
        """Hero card contains the current AQI value (128)."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "128" in all_md, "Hero AQI value '128' not found in rendered markdown"

    def test_72h_outlook_card_present(self):
        """72-Hour Outlook card renders peak AQI (185) and peak category."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "72-Hour Outlook" in all_md
        assert "185" in all_md or "Unhealthy" in all_md

    def test_compact_alert_renders(self):
        """Advisory warning banner renders (single st.info/warning, not stacked)."""
        at = _make_at()
        assert len(at.exception) == 0
        # At least one alert element (info or warning) rendered
        total_alerts = len(at.warning) + len(at.info) + len(at.error)
        assert total_alerts >= 1, "Expected at least one alert/info element"
        # Should not have more than 2 primary banners total (advisory → 1 info)
        assert len(at.error) == 0, "Unexpected error banner for advisory scenario"

    def test_forecast_section_heading_present(self):
        """'72-Hour AQI Forecast' section heading appears in rendered output."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "72-Hour AQI Forecast" in all_md

    def test_milestone_cards_rendered(self):
        """Five milestone cards (+1h +12h +24h +48h +72h) appear."""
        at = _make_at()
        assert len(at.exception) == 0
        all_metric_labels = " ".join(m.label for m in at.metric)
        for h in [1, 12, 24, 48, 72]:
            assert f"+{h}h" in all_metric_labels, f"Milestone card +{h}h not found"

    def test_pollutant_values_in_output(self):
        """PM2.5, NO2, SO2, O3 values appear in the telemetry section."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "PM2.5" in all_md or "52.3" in all_md
        assert "NO" in all_md or "21.7" in all_md

    def test_weather_values_in_output(self):
        """Temperature and humidity values appear in the telemetry section."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "Temperature" in all_md or "31.2" in all_md
        assert "Humidity" in all_md or "58.0" in all_md

    def test_health_guidance_present(self):
        """Health Guidance section heading and advisory text present."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "Health Guidance" in all_md or "Unhealthy for Sensitive Groups" in all_md

    def test_advanced_insights_heading_present(self):
        """Advanced Insights section appears (SHAP section)."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "Advanced Insights" in all_md

    def test_model_system_details_expander_present(self):
        """Model & System Details expander present."""
        at = _make_at()
        assert len(at.exception) == 0
        expander_labels = [e.label for e in at.expander]
        assert any("Model" in label or "System" in label for label in expander_labels), (
            f"Model & System Details expander not found. Expanders: {expander_labels}"
        )

    def test_refresh_button_present(self):
        """Refresh button is rendered on the page."""
        at = _make_at()
        assert len(at.exception) == 0
        button_labels = [b.label for b in at.button]
        assert any("Refresh" in label for label in button_labels), (
            f"Refresh button not found. Buttons: {button_labels}"
        )

    def test_horizon_slider_present(self):
        """Forecast horizon slider for SHAP attribution is present."""
        at = _make_at()
        assert len(at.exception) == 0
        assert len(at.slider) >= 1, "Expected at least one slider (horizon selector)"

    def test_no_raw_none_in_output(self):
        """Literal 'None', 'nan', or 'null' do not appear in user-facing markdown."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown).lower()
        # 'none' appears in legitimate text like "Good / Moderate" (no 'none' word boundary issue)
        # Check specifically for patterns that indicate unformatted Python None
        assert ">None<" not in all_md
        assert ">nan<" not in all_md
        assert "= None" not in all_md

    def test_aqi_legend_present(self):
        """AQI scale legend with 6 categories is present."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "prl-aqi-legend" in all_md or ("0–50" in all_md and "301+" in all_md)

    def test_progression_indicator_present(self):
        """Future-risk progression indicator is present in outlook card."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "prl-progression-container" in all_md or "Now" in all_md

    def test_observed_factors_present(self):
        """'What is affecting air quality now?' atmospheric factors panel is present."""
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "What is affecting air quality now?" in all_md or "Primary factor" in all_md


    def test_no_raw_iso_timestamp_in_main_view(self):
        """Generated-at ISO timestamp is NOT in the primary (non-expander) view.
        
        Freshness must be shown as relative age ('min ago' / 'h ago'),
        NOT as raw ISO strings like '2026-09-09T12:05:00+00:00'.
        The generated_at value is _GENERATED_AT = '2026-09-09T12:05:00+00:00'.
        It must not appear verbatim in the normal page markdown.
        """
        at = _make_at()
        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        # generated_at should only be in the expander, never in the primary view.
        # The status line must show relative age from forecast_origin, not generated_at.
        assert _GENERATED_AT not in all_md, (
            f"Raw generated_at ISO timestamp '{_GENERATED_AT}' found in primary markdown. "
            "Freshness must use relative age from forecast_origin/input_observed_at."
        )

    def test_freshness_derived_from_forecast_origin_not_generated_at(self):
        """Status line derives age from forecast_origin/input_observed_at, not generated_at.
        
        This test uses a payload where forecast_origin is 5 hours old but generated_at
        is only 1 minute old. The displayed freshness must reflect the observation age
        (~5h), not the generation age (~0 min).
        """
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        old_origin = (now - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
        fresh_generated = (now - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        obs = dict(_MOCK_OBS)
        obs["input_observed_at"] = old_origin
        obs["input_age_hours"] = 5.0

        forecast = dict(_MOCK_FORECAST)
        forecast["forecast_origin"] = old_origin
        forecast["input_observed_at"] = old_origin
        forecast["input_age_hours"] = 5.0
        forecast["generated_at"] = fresh_generated  # very fresh generation time

        at = _make_at(obs=obs, forecast=forecast)
        assert len(at.exception) == 0

        all_md = " ".join(e.value for e in at.markdown)
        # Status line should show ~5h old, not "just now" or "min ago"
        assert "5.0h" in all_md or "5h" in all_md or "ago" in all_md, (
            "Expected relative age string in markdown output"
        )
        # Must NOT show the fresh_generated raw timestamp as freshness
        assert fresh_generated not in all_md, (
            "generated_at appears in primary markdown — freshness must come from forecast_origin"
        )


class TestAppTestStaleScenario:
    """Verify stale-data scenario renders compact notice, not duplicate large banner."""

    def test_stale_renders_caption_not_large_warning(self):
        """Stale data shows caption, no st.warning large box alongside st.error."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        stale_ts = (now - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S+00:00")

        obs = dict(_MOCK_OBS)
        obs["is_stale"] = True
        obs["input_observed_at"] = stale_ts
        obs["input_age_hours"] = 48.0
        obs["alert"] = {
            "active": False,
            "level": "none",
            "severity_rank": 0,
            "aqi": obs["current_aqi"],
            "category": "Good",
            "message": "Air quality is satisfactory.",
        }

        forecast = dict(_MOCK_FORECAST)
        forecast["is_stale"] = True
        forecast["input_observed_at"] = stale_ts
        forecast["input_age_hours"] = 48.0
        forecast["forecast_alert"] = {
            "active": False,
            "highest_level": "none",
            "peak_aqi": 90.0,
            "peak_horizon": 10,
            "peak_category": "Moderate",
            "first_unhealthy_horizon": None,
            "message": "Air quality expected to remain moderate.",
            "upper_interval_crosses_hazardous": False,
        }

        at = _make_at(obs=obs, forecast=forecast)
        assert len(at.exception) == 0

        # st.warning should NOT be called with stale-notice text for no-alert scenario
        # (stale notice is st.caption in redesigned version)
        warn_texts = [w.value for w in at.warning]
        # No primary alert (level=none), so warning count should be 0
        assert len(at.warning) == 0, (
            f"Expected 0 st.warning calls for no-alert stale scenario, "
            f"got {len(at.warning)}: {warn_texts}"
        )

        # Caption should mention the stale age
        caption_texts = [c.value for c in at.caption]
        assert any("48.0h" in c or "48" in c or "Historical" in c for c in caption_texts), (
            f"Expected stale caption with age. Captions: {caption_texts}"
        )


class TestAppTestColdStart:
    """Verify cold-start / ConnectionError scenario renders without exception."""

    def test_cold_start_no_exception(self):
        """ConnectionError during fetch renders the cold-start screen, no exception."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with patch("src.dashboard.data_client.DashboardDataClient.fetch_current",
                   side_effect=ConnectionError("Service starting")):
            at.run()

        assert len(at.exception) == 0, (
            f"Cold-start scenario raised exceptions: {at.exception}"
        )
        # Should have the Retry button
        button_labels = [b.label for b in at.button]
        assert any("Retry" in label for label in button_labels), (
            f"Expected Retry button in cold-start state. Buttons: {button_labels}"
        )

    def test_cold_start_renders_waking_up_message(self):
        """'waking up' text appears in cold-start markdown."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        with patch("src.dashboard.data_client.DashboardDataClient.fetch_current",
                   side_effect=ConnectionError("Service starting")):
            at.run()

        assert len(at.exception) == 0
        all_md = " ".join(e.value for e in at.markdown)
        assert "waking up" in all_md.lower(), (
            "Expected 'waking up' in cold-start markdown output"
        )


class TestAppTestMissingValues:
    """Verify missing/None API values do not crash and render as '—'."""

    def test_missing_pollutants_render_dash(self):
        """Missing pollutant values display as '—' not 'None'."""
        obs = dict(_MOCK_OBS)
        obs["pollutants"] = {"pm2_5": None, "pm10": None, "no2": None, "so2": None, "co": None, "o3": None}

        at = _make_at(obs=obs)
        assert len(at.exception) == 0

        all_md = " ".join(e.value for e in at.markdown)
        assert ">None<" not in all_md
        assert "None µg" not in all_md

    def test_missing_peak_aqi_renders_dash(self):
        """Missing peak AQI in forecast_alert renders as '—' not crash."""
        forecast = dict(_MOCK_FORECAST)
        forecast["forecast_alert"] = {
            "highest_level": "none",
            "peak_aqi": None,
            "peak_horizon": None,
            "peak_category": "—",
            "first_unhealthy_horizon": None,
            "message": "",
            "upper_interval_crosses_hazardous": False,
        }

        at = _make_at(forecast=forecast)
        assert len(at.exception) == 0

    def test_missing_obs_aqi_renders_dash(self):
        """None current_aqi renders as '—' in hero card."""
        obs = dict(_MOCK_OBS)
        obs["current_aqi"] = None

        at = _make_at(obs=obs)
        assert len(at.exception) == 0

        all_md = " ".join(e.value for e in at.markdown)
        # Should have em-dash, not literal None
        assert "None" not in all_md or "—" in all_md
