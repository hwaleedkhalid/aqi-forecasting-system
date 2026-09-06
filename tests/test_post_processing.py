"""Unit tests for src.inference.post_processing."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.inference.post_processing import AQIPostProcessor


class TestAQIPostProcessor:
    """Test suite for AQIPostProcessor."""

    def test_loads_empirical_error_intervals(self):
        processor = AQIPostProcessor()
        assert len(processor.q10) == 72
        assert len(processor.q90) == 72
        # Error envelope should be non-trivial
        assert all(q10 < q90 for q10, q90 in zip(processor.q10, processor.q90))

    def test_compute_empirical_error_bounds_bounds_lower_at_zero(self):
        processor = AQIPostProcessor()
        # With very low AQI (e.g. 5.0) and negative q10, lower bound must clamp to 0.0
        lower, upper = processor.compute_empirical_error_bounds(1, 5.0)
        assert lower >= 0.0
        assert upper >= lower

    def test_process_horizon_point_categorizes_all_tiers_correctly(self):
        processor = AQIPostProcessor()
        now = datetime.now(timezone.utc)

        cases = [
            (25.0, "Good", "#00E400", False, False),
            (75.0, "Moderate", "#FFFF00", False, False),
            (125.0, "Unhealthy for Sensitive Groups", "#FF7E00", False, False),
            (175.0, "Unhealthy", "#FF0000", False, False),
            (250.0, "Very Unhealthy", "#8F3F97", True, False),
            (350.0, "Hazardous", "#7E0023", True, True),
            (550.0, "Hazardous", "#7E0023", True, True),  # Extreme value above 500
        ]

        for aqi_val, expected_cat, expected_col, exp_high_sev, exp_haz in cases:
            point = processor.process_horizon_point(1, now, aqi_val)
            assert point["aqi"] == aqi_val
            assert point["category"] == expected_cat
            assert point["color"] == expected_col
            assert point["high_severity"] == exp_high_sev
            assert point["hazardous"] == exp_haz
            assert len(point["health_advisory"]) > 10

    def test_process_horizon_preserves_raw_aqi_above_500(self):
        """Do not automatically clamp AQI prediction value at 500."""
        processor = AQIPostProcessor()
        now = datetime.now(timezone.utc)
        point = processor.process_horizon_point(12, now, 620.5)
        assert point["aqi"] == 620.5
        assert point["category"] == "Hazardous"
        assert point["hazardous"] is True

    def test_process_72h_forecast_produces_72_points_with_incremental_timestamps(self):
        processor = AQIPostProcessor()
        origin = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)
        raw_preds = [100.0 + i for i in range(72)]

        forecasts = processor.process_72h_forecast(raw_preds, forecast_origin=origin)
        assert len(forecasts) == 72

        for i, f in enumerate(forecasts):
            assert f["horizon"] == i + 1
            assert "forecast_time" in f
            assert f["error_lower"] <= f["error_upper"]

        # Check timestamp progression
        t1 = datetime.fromisoformat(forecasts[0]["forecast_time"])
        t72 = datetime.fromisoformat(forecasts[71]["forecast_time"])
        diff_hours = (t72 - t1).total_seconds() / 3600.0
        assert diff_hours == 71.0

    def test_process_72h_forecast_raises_on_invalid_length(self):
        processor = AQIPostProcessor()
        with pytest.raises(ValueError, match="Expected 72"):
            processor.process_72h_forecast([50.0] * 50)

    def test_build_summary_detects_peak_and_alerts(self):
        processor = AQIPostProcessor()
        origin = datetime.now(timezone.utc)
        raw_preds = [50.0] * 72
        raw_preds[23] = 350.0  # Peak hazardous at h=24
        raw_preds[47] = 220.0  # High severity at h=48

        forecasts = processor.process_72h_forecast(raw_preds, forecast_origin=origin)
        summary = processor.build_summary(forecasts)

        assert summary["peak_aqi"] == 350.0
        assert summary["peak_horizon"] == 24
        assert summary["peak_category"] == "Hazardous"
        assert summary["has_high_severity"] is True
        assert summary["has_hazardous"] is True
        assert summary["highest_alert_level"] == "Hazardous (>300)"
