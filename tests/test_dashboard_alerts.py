"""Unit tests for Dashboard Alert Presentation Components.

Verifies the redesigned compact alert system:
1. Single primary banner (highest severity — current or forecast)
2. Stale compact caption (not large banner)
3. Uncertainty collapsed expander
4. No stacking of multiple large alert boxes

Also verifies:
- Plotly reference lines at EPA category boundaries (151, 201, 301)
- render_freshness_banner legacy compatibility
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.dashboard.components import (
    build_forecast_figure,
    render_alert_banners,
    render_freshness_banner,
    render_sidebar,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_obs(level: str, aqi: float = 120.0, is_stale: bool = False) -> dict:
    category_map = {
        "none": "Good",
        "advisory": "Unhealthy for Sensitive Groups",
        "warning": "Unhealthy",
        "severe": "Very Unhealthy",
        "hazardous": "Hazardous",
    }
    return {
        "current_aqi": aqi,
        "category": category_map.get(level, "Unknown"),
        "is_stale": is_stale,
        "input_observed_at": "2026-09-08T00:00:00+00:00",
        "input_age_hours": 24.5 if is_stale else 0.3,
        "alert": {
            "active": level != "none",
            "level": level,
            "severity_rank": {"none": 0, "advisory": 2, "warning": 3, "severe": 4, "hazardous": 5}.get(level, 0),
            "category": category_map.get(level, "Unknown"),
            "aqi": aqi,
            "message": f"Alert at level {level}.",
        },
    }


def _make_forecast_alert(level: str, peak_aqi: float = 200.0, is_stale: bool = False, upper_crosses: bool = False) -> dict:
    return {
        "is_stale": is_stale,
        "input_observed_at": "2026-09-08T00:00:00+00:00",
        "input_age_hours": 24.5 if is_stale else 0.3,
        "forecast_alert": {
            "active": level != "none",
            "highest_level": level,
            "peak_aqi": peak_aqi,
            "peak_horizon": 18,
            "message": f"Forecast alert {level}.",
            "upper_interval_crosses_hazardous": upper_crosses,
        },
    }


# ── Primary banner tests ──────────────────────────────────────────────────────

class TestPrimaryAlertBanner:
    """Verify that only ONE primary banner renders regardless of combined severity."""

    def test_current_hazardous_renders_single_error(self):
        obs = _make_obs("hazardous", aqi=340.0)
        forecast = _make_forecast_alert("severe", peak_aqi=245.0)
        with patch("streamlit.error") as mock_err, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.caption"):
            render_alert_banners(obs, forecast)
            # Only one primary error banner (current hazardous wins, it's highest)
            assert mock_err.call_count == 1
            msg = mock_err.call_args[0][0]
            assert "Hazardous" in msg
            assert "340" in msg
            mock_warn.assert_not_called()

    def test_forecast_severe_renders_single_error(self):
        obs = _make_obs("none", aqi=75.0)
        forecast = _make_forecast_alert("severe", peak_aqi=245.0)
        with patch("streamlit.error") as mock_err, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.caption"):
            render_alert_banners(obs, forecast)
            assert mock_err.call_count == 1
            assert "Very Unhealthy" in mock_err.call_args[0][0]
            mock_warn.assert_not_called()

    def test_warning_tier_renders_single_warning(self):
        obs = _make_obs("warning", aqi=165.0)
        forecast = _make_forecast_alert("warning", peak_aqi=180.0)
        with patch("streamlit.error") as mock_err, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.caption"):
            render_alert_banners(obs, forecast)
            mock_err.assert_not_called()
            assert mock_warn.call_count == 1
            assert "Unhealthy" in mock_warn.call_args[0][0]

    def test_advisory_tier_renders_single_info(self):
        obs = _make_obs("advisory", aqi=120.0)
        forecast = _make_forecast_alert("advisory", peak_aqi=130.0)
        with patch("streamlit.error") as mock_err, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.caption"):
            render_alert_banners(obs, forecast)
            mock_err.assert_not_called()
            mock_warn.assert_not_called()
            assert mock_info.call_count == 1

    def test_no_alert_no_banner(self):
        obs = _make_obs("none", aqi=45.0)
        forecast = _make_forecast_alert("none", peak_aqi=55.0)
        with patch("streamlit.error") as mock_err, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.info") as mock_info, \
             patch("streamlit.caption"):
            render_alert_banners(obs, forecast)
            mock_err.assert_not_called()
            mock_warn.assert_not_called()
            mock_info.assert_not_called()


# ── Stale notice tests ────────────────────────────────────────────────────────

class TestStaleNotice:
    """Verify stale data renders as compact caption, not large warning banner."""

    def test_stale_severe_renders_error_plus_caption(self):
        obs = _make_obs("none", is_stale=True)
        forecast = _make_forecast_alert("severe", is_stale=True)
        with patch("streamlit.error") as mock_err, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.caption") as mock_cap, \
             patch("streamlit.info"):
            render_alert_banners(obs, forecast)
            # Primary: severe error banner
            mock_err.assert_called_once()
            # Stale: compact caption (not a large warning)
            mock_cap.assert_called_once()
            stale_text = mock_cap.call_args[0][0]
            assert "24.5h" in stale_text
            mock_warn.assert_not_called()  # No big yellow stale box

    def test_stale_only_no_primary_renders_caption(self):
        obs = _make_obs("none", is_stale=True)
        forecast = _make_forecast_alert("none", is_stale=True)
        with patch("streamlit.error") as mock_err, \
             patch("streamlit.warning") as mock_warn, \
             patch("streamlit.caption") as mock_cap, \
             patch("streamlit.info"):
            render_alert_banners(obs, forecast)
            mock_err.assert_not_called()
            mock_warn.assert_not_called()
            mock_cap.assert_called_once()


# ── Uncertainty expander test ─────────────────────────────────────────────────

class TestUncertaintyExpander:
    """Verify upper-interval-crosses-hazardous uses collapsed expander."""

    def test_uncertainty_expander_rendered(self):
        obs = _make_obs("none", aqi=80.0)
        forecast = _make_forecast_alert("none", upper_crosses=True)
        exp_mock = MagicMock()
        exp_mock.__enter__ = lambda s: s
        exp_mock.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.expander", return_value=exp_mock) as mock_exp, \
             patch("streamlit.error"), \
             patch("streamlit.warning"), \
             patch("streamlit.info"), \
             patch("streamlit.caption"):
            render_alert_banners(obs, forecast)
            # Expander should be called with uncertainty title
            exp_calls = [str(c) for c in mock_exp.call_args_list]
            assert any("uncertainty" in c.lower() for c in exp_calls)

    def test_uncertainty_expander_not_rendered_when_already_hazardous(self):
        """No uncertainty expander if highest level is already hazardous."""
        obs = _make_obs("hazardous", aqi=320.0)
        forecast = _make_forecast_alert("hazardous", peak_aqi=340.0, upper_crosses=True)
        exp_mock = MagicMock()
        exp_mock.__enter__ = lambda s: s
        exp_mock.__exit__ = MagicMock(return_value=False)
        with patch("streamlit.expander", return_value=exp_mock) as mock_exp, \
             patch("streamlit.error"), patch("streamlit.warning"), \
             patch("streamlit.info"), patch("streamlit.caption"):
            render_alert_banners(obs, forecast)
            # Uncertainty expander not called when already hazardous (per priority rule)
            exp_calls = [str(c) for c in mock_exp.call_args_list]
            uncertainty_called = any("uncertainty" in c.lower() for c in exp_calls)
            assert not uncertainty_called


# ── Freshness banner legacy compatibility ─────────────────────────────────────

class TestFreshnessLegacy:
    """Legacy render_freshness_banner still works (called internally)."""

    def test_freshness_banner_stale(self):
        with patch("streamlit.warning") as mock_warn:
            render_freshness_banner(
                is_stale=True,
                observed_at="2026-08-31T07:00:00+00:00",
                age_hours=145.0,
            )
            mock_warn.assert_called_once()
            assert "145.0 hours" in mock_warn.call_args[0][0]
            assert "Telemetry Notice" in mock_warn.call_args[0][0]

    def test_freshness_banner_not_stale(self):
        with patch("streamlit.warning") as mock_warn:
            render_freshness_banner(
                is_stale=False,
                observed_at="2026-09-09T12:00:00+00:00",
                age_hours=0.3,
            )
            mock_warn.assert_not_called()


# ── Sidebar no-op test ────────────────────────────────────────────────────────

class TestSidebarNoOp:
    """render_sidebar is a no-op stub — sidebar is permanently removed."""

    def test_sidebar_noop_no_exception(self):
        render_sidebar({}, {}, "REST API")  # Must not raise

    def test_sidebar_writes_nothing(self):
        with patch("streamlit.sidebar") as mock_sidebar:
            render_sidebar({"model_id": "EXP-019"}, {"peak_aqi": 100}, "REST API")
            mock_sidebar.assert_not_called()


# ── EPA reference lines (unchanged) ──────────────────────────────────────────

class TestForecastChartReferenceLines:
    """Verify exact EPA reference lines at 151, 201, and 301."""

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
