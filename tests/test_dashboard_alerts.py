"""Unit tests for Dashboard Alert Presentation Components.

Verifies priority hierarchy:
1. Current Severe/Hazardous
2. Forecast Severe/Hazardous
3. Stale Data Notice
4. Warnings and Advisories
5. Upper Empirical Bound Crossings

Also verifies Plotly reference lines at EPA category boundaries (151, 201, 301)
and sidebar forecast alert rendering.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.dashboard.components import (
    build_forecast_figure,
    render_alert_banners,
    render_sidebar,
)


class TestDashboardAlertBanners:
    """Test priority rendering and message formatting in render_alert_banners."""

    def test_current_hazardous_banner_priority(self):
        """Current Hazardous observation renders top emergency error banner."""
        obs_data = {
            "current_aqi": 340.0,
            "category": "Hazardous",
            "is_stale": False,
            "alert": {
                "active": True,
                "level": "hazardous",
                "severity_rank": 5,
                "category": "Hazardous",
                "aqi": 340.0,
                "message": "Health warning of emergency conditions.",
            },
        }
        forecast_data = {
            "is_stale": False,
            "forecast_alert": {
                "active": True,
                "highest_level": "severe",
                "message": "The model forecasts Very Unhealthy conditions.",
            },
        }

        with patch("streamlit.error") as mock_error, patch("streamlit.warning") as mock_warn:
            render_alert_banners(obs_data, forecast_data)
            # Both Current Hazardous and Forecast Severe error banners should render
            assert mock_error.call_count == 2
            first_call_msg = mock_error.call_args_list[0][0][0]
            assert "HAZARDOUS AIR QUALITY EMERGENCY" in first_call_msg
            assert "340 AQI" in first_call_msg

    def test_forecast_severe_with_stale_notice(self):
        """Stale severe forecast displays BOTH severe alert and stale notice."""
        obs_data = {
            "current_aqi": 75.0,
            "category": "Moderate",
            "is_stale": True,
            "input_observed_at": "2026-09-08T00:00:00+00:00",
            "input_age_hours": 24.5,
            "alert": {
                "active": False,
                "level": "none",
                "severity_rank": 1,
                "category": "Moderate",
                "aqi": 75.0,
                "message": "Air quality is acceptable.",
            },
        }
        forecast_data = {
            "is_stale": True,
            "input_observed_at": "2026-09-08T00:00:00+00:00",
            "input_age_hours": 24.5,
            "forecast_alert": {
                "active": True,
                "highest_level": "severe",
                "message": "The model forecasts Very Unhealthy air quality conditions within the next 72 hours (Peak AQI 245.0 at +18h).",
            },
        }

        with patch("streamlit.error") as mock_error, patch("streamlit.warning") as mock_warn:
            render_alert_banners(obs_data, forecast_data)
            # Severe forecast banner rendered
            mock_error.assert_called_once()
            assert "VERY UNHEALTHY AQI FORECAST" in mock_error.call_args[0][0]
            # Stale notice rendered
            mock_warn.assert_called_once()
            assert "Telemetry Notice" in mock_warn.call_args[0][0]
            assert "24.5 hours" in mock_warn.call_args[0][0]

    def test_warning_tier_rendered_when_no_severe(self):
        """Unhealthy warning (151-200) renders warning banner."""
        obs_data = {
            "current_aqi": 165.0,
            "category": "Unhealthy",
            "is_stale": False,
            "alert": {
                "active": True,
                "level": "warning",
                "severity_rank": 3,
                "category": "Unhealthy",
                "aqi": 165.0,
                "message": "Some members of general public may experience effects.",
            },
        }
        forecast_data = {
            "is_stale": False,
            "forecast_alert": {
                "active": True,
                "highest_level": "warning",
                "message": "The model forecasts Unhealthy air quality conditions within the next 72 hours.",
            },
        }

        with patch("streamlit.error") as mock_error, patch("streamlit.warning") as mock_warn:
            render_alert_banners(obs_data, forecast_data)
            mock_error.assert_not_called()
            mock_warn.assert_called_once()
            assert "UNHEALTHY AIR QUALITY WARNING" in mock_warn.call_args[0][0]

    def test_upper_interval_crossing_hazardous_uncertainty_banner(self):
        """When point forecast is warning/moderate but error upper bound touches >300."""
        obs_data = {
            "current_aqi": 80.0,
            "category": "Moderate",
            "is_stale": False,
            "alert": {"active": False, "level": "none", "severity_rank": 1, "aqi": 80.0},
        }
        forecast_data = {
            "is_stale": False,
            "forecast_alert": {
                "active": False,
                "highest_level": "none",
                "upper_interval_crosses_hazardous": True,
            },
        }

        with patch("streamlit.info") as mock_info:
            render_alert_banners(obs_data, forecast_data)
            mock_info.assert_called_once()
            assert "Uncertainty Notice" in mock_info.call_args[0][0]
            assert "Hazardous threshold (>300 AQI)" in mock_info.call_args[0][0]


class TestSidebarAlertIntegration:
    """Verify render_sidebar consumes forecast_alert contract."""

    def test_sidebar_hazardous_alert(self):
        summary = {
            "peak_aqi": 320.5,
            "peak_horizon": 14,
            "peak_category": "Hazardous",
            "forecast_alert": {
                "highest_level": "hazardous",
                "first_hazardous_horizon": 12,
            },
        }
        with patch("streamlit.sidebar.error") as mock_err:
            render_sidebar({}, summary, "REST API")
            mock_err.assert_called_once()
            assert "Hazardous AQI Emergency" in mock_err.call_args[0][0]
            assert "+12h" in mock_err.call_args[0][0]

    def test_sidebar_normal_condition(self):
        summary = {
            "peak_aqi": 85.0,
            "peak_horizon": 4,
            "peak_category": "Moderate",
            "forecast_alert": {
                "highest_level": "none",
            },
        }
        with patch("streamlit.sidebar.success") as mock_succ:
            render_sidebar({}, summary, "REST API")
            mock_succ.assert_called_once()
            assert "Good or Moderate" in mock_succ.call_args[0][0]


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
