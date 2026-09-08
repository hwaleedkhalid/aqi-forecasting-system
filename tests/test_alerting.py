"""Unit tests for the centralized AQI Alerting and Severity Evaluator.

Validates single source of truth category classification, boundary conditions (EPA AQI 0..650),
multi-horizon forecast scanning, threshold crossings, empirical error bounds, and provenance propagation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from src.feature_pipeline.aqi_calculator import get_aqi_category
from src.inference.alerting import (
    CATEGORY_TO_ALERT_CONFIG,
    AQIAlert,
    ForecastAlertSummary,
    classify_category_alert,
    evaluate_current_alert,
    evaluate_forecast_alerts,
)


class TestCategoryAlertClassification:
    """Test mapping from EPA categories to normalized alert levels and severity ranks."""

    def test_all_epa_categories_mapped_correctly(self):
        expected = {
            "Good": ("none", 0, False, None),
            "Moderate": ("none", 1, False, None),
            "Unhealthy for Sensitive Groups": ("advisory", 2, True, 101.0),
            "Unhealthy": ("warning", 3, True, 151.0),
            "Very Unhealthy": ("severe", 4, True, 201.0),
            "Hazardous": ("hazardous", 5, True, 301.0),
        }
        for cat, (level, rank, active, thresh) in expected.items():
            cfg = classify_category_alert(cat)
            assert cfg["level"] == level, f"Mismatch for {cat} level"
            assert cfg["severity_rank"] == rank, f"Mismatch for {cat} rank"
            assert cfg["active"] is active, f"Mismatch for {cat} active"
            assert cfg["threshold"] == thresh, f"Mismatch for {cat} threshold"

    def test_unknown_category_fallback(self):
        cfg = classify_category_alert("NonExistentCategory")
        assert cfg["level"] == "none"
        assert cfg["severity_rank"] == 0
        assert cfg["active"] is False


class TestEPABoundariesAndSingleSourceOfTruth:
    """Verify exact EPA boundary points through the single source of truth path:
    numeric AQI -> get_aqi_category() -> category -> classify_category_alert().
    """

    @pytest.mark.parametrize(
        "aqi_val, expected_cat, expected_level, expected_rank, expected_active",
        [
            (0.0, "Good", "none", 0, False),
            (25.0, "Good", "none", 0, False),
            (50.0, "Good", "none", 0, False),
            (51.0, "Moderate", "none", 1, False),
            (75.0, "Moderate", "none", 1, False),
            (100.0, "Moderate", "none", 1, False),
            (101.0, "Unhealthy for Sensitive Groups", "advisory", 2, True),
            (125.0, "Unhealthy for Sensitive Groups", "advisory", 2, True),
            (150.0, "Unhealthy for Sensitive Groups", "advisory", 2, True),
            (151.0, "Unhealthy", "warning", 3, True),
            (175.0, "Unhealthy", "warning", 3, True),
            (200.0, "Unhealthy", "warning", 3, True),
            (201.0, "Very Unhealthy", "severe", 4, True),
            (250.0, "Very Unhealthy", "severe", 4, True),
            (300.0, "Very Unhealthy", "severe", 4, True),
            (301.0, "Hazardous", "hazardous", 5, True),
            (400.0, "Hazardous", "hazardous", 5, True),
            (500.0, "Hazardous", "hazardous", 5, True),
            (501.0, "Hazardous", "hazardous", 5, True),  # Extreme values preserved
            (650.0, "Hazardous", "hazardous", 5, True),  # Unclipped extreme
        ],
    )
    def test_boundary_points(self, aqi_val, expected_cat, expected_level, expected_rank, expected_active):
        alert = evaluate_current_alert(aqi_val)
        assert alert.category == expected_cat
        assert alert.level == expected_level
        assert alert.severity_rank == expected_rank
        assert alert.active is expected_active
        assert alert.aqi == round(aqi_val, 1)

    def test_rounding_boundary_consistency(self):
        """EPA category calculation rounds float to nearest integer."""
        # 50.4 rounds to 50 -> Good
        alert_low = evaluate_current_alert(50.4)
        assert alert_low.category == "Good"
        assert alert_low.level == "none"

        # 50.6 rounds to 51 -> Moderate
        alert_high = evaluate_current_alert(50.6)
        assert alert_high.category == "Moderate"
        assert alert_high.level == "none"

        # 200.4 rounds to 200 -> Unhealthy
        alert_200 = evaluate_current_alert(200.4)
        assert alert_200.category == "Unhealthy"
        assert alert_200.level == "warning"

        # 200.6 rounds to 201 -> Very Unhealthy
        alert_201 = evaluate_current_alert(200.6)
        assert alert_201.category == "Very Unhealthy"
        assert alert_201.level == "severe"


class TestCurrentAlertEvaluation:
    """Test evaluate_current_alert metadata and provenance fields."""

    def test_provenance_and_staleness_propagation(self):
        alert = evaluate_current_alert(
            aqi=185.2,
            data_is_stale=True,
            feature_source="hopsworks",
            fallback_active=False,
        )
        d = alert.to_dict()
        assert d["aqi"] == 185.2
        assert d["category"] == "Unhealthy"
        assert d["level"] == "warning"
        assert d["severity_rank"] == 3
        assert d["active"] is True
        assert d["threshold"] == 151.0
        assert d["data_is_stale"] is True
        assert d["feature_source"] == "hopsworks"
        assert d["fallback_active"] is False
        assert "Unhealthy" in d["title"]
        assert len(d["message"]) > 0

    def test_fallback_active_flag(self):
        alert = evaluate_current_alert(
            aqi=72.0,
            data_is_stale=False,
            feature_source="bootstrap",
            fallback_active=True,
        )
        assert alert.fallback_active is True
        assert alert.feature_source == "bootstrap"
        assert alert.active is False  # Moderate is not active alert


class TestForecastAlertEvaluation:
    """Test evaluate_forecast_alerts across various 72-hour sequence patterns."""

    def _make_dummy_forecasts(self, aqi_list: list[float], upper_offset: float = 20.0) -> list[dict]:
        base_time = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
        forecasts = []
        for h, aqi in enumerate(aqi_list, start=1):
            cat, color = get_aqi_category(aqi)
            cfg = classify_category_alert(cat)
            forecasts.append({
                "horizon": h,
                "forecast_time": base_time.isoformat(),
                "aqi": round(aqi, 1),
                "category": cat,
                "color": color,
                "error_lower": max(0.0, aqi - 20.0),
                "error_upper": aqi + upper_offset,
                "high_severity": bool(aqi > 200.0),
                "hazardous": bool(aqi > 300.0),
                "alert_level": cfg["level"],
                "severity_rank": cfg["severity_rank"],
            })
        return forecasts

    def test_empty_forecast_list(self):
        res = evaluate_forecast_alerts([])
        assert res.active is False
        assert res.highest_level == "none"
        assert res.highest_severity_rank == 0
        assert res.peak_aqi == 0.0

    def test_all_good_and_moderate(self):
        # All 72 horizons under 100
        aqis = [45.0] * 36 + [75.0] * 36
        forecasts = self._make_dummy_forecasts(aqis)
        res = evaluate_forecast_alerts(forecasts)

        assert res.active is False
        assert res.highest_level == "none"
        assert res.highest_severity_rank == 1
        assert res.peak_aqi == 75.0
        assert res.advisory_horizon_count == 0
        assert res.unhealthy_horizon_count == 0
        assert res.very_unhealthy_horizon_count == 0
        assert res.hazardous_horizon_count == 0
        assert res.first_advisory_horizon is None

    def test_peak_in_advisory_range(self):
        # 1..20: 80 AQI, 21..30: 120 AQI, 31..72: 90 AQI
        aqis = [80.0] * 20 + [120.0] * 10 + [90.0] * 42
        forecasts = self._make_dummy_forecasts(aqis)
        res = evaluate_forecast_alerts(forecasts)

        assert res.active is True
        assert res.highest_level == "advisory"
        assert res.highest_severity_rank == 2
        assert res.peak_aqi == 120.0
        assert res.peak_horizon == 21
        assert res.first_advisory_horizon == 21
        assert res.advisory_horizon_count == 10
        assert res.first_unhealthy_horizon is None
        assert res.unhealthy_horizon_count == 0

    def test_peak_in_unhealthy_warning_range(self):
        # Enters advisory at h10, enters warning at h15, peaks at 185 at h20
        aqis = [80.0] * 9 + [120.0] * 5 + [170.0] * 5 + [185.0] + [160.0] * 10 + [90.0] * 42
        forecasts = self._make_dummy_forecasts(aqis)
        res = evaluate_forecast_alerts(forecasts)

        assert res.active is True
        assert res.highest_level == "warning"
        assert res.highest_severity_rank == 3
        assert res.peak_aqi == 185.0
        assert res.peak_horizon == 20
        assert res.first_advisory_horizon == 10
        assert res.first_unhealthy_horizon == 15
        assert res.unhealthy_horizon_count == 16
        assert res.advisory_horizon_count == 5
        assert res.first_very_unhealthy_horizon is None

    def test_peak_in_severe_range(self):
        # Reaches Very Unhealthy (201-300) at h24, peaks at 260
        aqis = [100.0] * 10 + [160.0] * 13 + [240.0] * 5 + [260.0] + [180.0] * 43
        forecasts = self._make_dummy_forecasts(aqis)
        res = evaluate_forecast_alerts(forecasts)

        assert res.active is True
        assert res.highest_level == "severe"
        assert res.highest_severity_rank == 4
        assert res.peak_aqi == 260.0
        assert res.peak_horizon == 29
        assert res.first_very_unhealthy_horizon == 24
        assert res.very_unhealthy_horizon_count == 6
        assert res.severe_or_higher_horizon_count == 6
        assert res.first_hazardous_horizon is None
        assert res.hazardous_horizon_count == 0

    def test_peak_in_hazardous_range_and_multi_crossing(self):
        # Progresses through all levels:
        # h1-5: 80 (Moderate)
        # h6-10: 120 (Advisory)
        # h11-20: 180 (Warning)
        # h21-30: 250 (Severe)
        # h31-40: 330 (Hazardous)
        # h41-72: 90 (Moderate)
        aqis = [80.0] * 5 + [120.0] * 5 + [180.0] * 10 + [250.0] * 10 + [330.0] * 10 + [90.0] * 32
        forecasts = self._make_dummy_forecasts(aqis)
        res = evaluate_forecast_alerts(forecasts)

        assert res.active is True
        assert res.highest_level == "hazardous"
        assert res.highest_severity_rank == 5
        assert res.peak_aqi == 330.0
        assert res.first_advisory_horizon == 6
        assert res.first_unhealthy_horizon == 11
        assert res.first_very_unhealthy_horizon == 21
        assert res.first_hazardous_horizon == 31
        assert res.hazardous_horizon_count == 10
        assert res.very_unhealthy_horizon_count == 10
        assert res.unhealthy_horizon_count == 10
        assert res.advisory_horizon_count == 5
        assert res.severe_or_higher_horizon_count == 20

    def test_upper_interval_crosses_hazardous_detection(self):
        # Forecast point is 180 (Unhealthy), but residual upper bound reaches 315 (>300)
        aqis = [180.0] * 72
        forecasts = self._make_dummy_forecasts(aqis, upper_offset=135.0)  # 180 + 135 = 315
        res = evaluate_forecast_alerts(forecasts)

        assert res.highest_level == "warning"
        assert res.upper_interval_crosses_hazardous is True

    def test_stale_and_fallback_provenance(self):
        aqis = [220.0] * 72
        forecasts = self._make_dummy_forecasts(aqis)
        res = evaluate_forecast_alerts(
            forecasts,
            data_is_stale=True,
            fallback_active=True,
            feature_source="bootstrap",
        )

        assert res.highest_level == "severe"
        assert res.data_is_stale is True
        assert res.fallback_active is True
        assert res.feature_source == "bootstrap"
        assert "The model forecasts Very Unhealthy" in res.message
