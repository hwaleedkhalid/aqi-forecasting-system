"""Unit tests for US EPA AQI Calculator.

Validates piecewise linear interpolation against EPA standard breakpoints,
dominant pollutant identification, sentinel/missing value safety, and category mapping.
"""

import numpy as np
import pytest

from src.feature_pipeline.aqi_calculator import (
    calculate_overall_aqi,
    calculate_sub_index,
    get_aqi_category,
    truncate_concentration,
)


# =============================================================================
# Concentration Truncation Tests
# =============================================================================

class TestConcentrationTruncation:
    """Test standard EPA rounding/truncation rules."""

    def test_pm25_truncation(self) -> None:
        assert truncate_concentration("pm2_5", 12.345) == 12.3
        assert truncate_concentration("pm2_5", 35.49) == 35.5

    def test_other_pollutants(self) -> None:
        assert truncate_concentration("pm10", 154.2) == 154.2
        assert truncate_concentration("unknown_pol", 10.5) == 10.5


# =============================================================================
# Sub-Index Calculation Tests
# =============================================================================

class TestSubIndexCalculation:
    """Test EPA breakpoint linear interpolation across all criteria pollutants."""

    def test_pm25_boundary_values(self) -> None:
        assert calculate_sub_index("pm2_5", 0.0) == 0
        assert calculate_sub_index("pm2_5", 9.0) == 50
        assert calculate_sub_index("pm2_5", 9.1) == 51
        assert calculate_sub_index("pm2_5", 35.4) == 100
        assert calculate_sub_index("pm2_5", 35.5) == 101
        assert calculate_sub_index("pm2_5", 55.4) == 150
        assert calculate_sub_index("pm2_5", 55.5) == 151
        assert calculate_sub_index("pm2_5", 125.4) == 200
        assert calculate_sub_index("pm2_5", 125.5) == 201
        assert calculate_sub_index("pm2_5", 225.4) == 300
        assert calculate_sub_index("pm2_5", 225.5) == 301
        assert calculate_sub_index("pm2_5", 325.4) == 400
        assert calculate_sub_index("pm2_5", 325.5) == 401
        assert calculate_sub_index("pm2_5", 500.4) == 500

    def test_pm25_extreme_values(self) -> None:
        # Above maximum defined EPA breakpoint should clamp at 500
        assert calculate_sub_index("pm2_5", 650.0) == 500

    def test_pm10_breakpoints(self) -> None:
        assert calculate_sub_index("pm10", 0.0) == 0
        assert calculate_sub_index("pm10", 54.0) == 50
        assert calculate_sub_index("pm10", 154.0) == 100
        assert calculate_sub_index("pm10", 254.0) == 150
        assert calculate_sub_index("pm10", 354.0) == 200
        assert calculate_sub_index("pm10", 424.0) == 300
        assert calculate_sub_index("pm10", 504.0) == 400
        assert calculate_sub_index("pm10", 604.0) == 500
        assert calculate_sub_index("pm10", 800.0) == 500

    def test_other_criteria_pollutants(self) -> None:
        # O3
        assert calculate_sub_index("o3", 50.0) <= 50
        assert calculate_sub_index("o3", 120.0) == pytest.approx(71, abs=3)
        # NO2
        assert calculate_sub_index("no2", 50.0) <= 50
        assert calculate_sub_index("no2", 150.0) == pytest.approx(79, abs=3)
        # SO2
        assert calculate_sub_index("so2", 40.0) <= 50
        assert calculate_sub_index("so2", 150.0) == pytest.approx(78, abs=3)
        # CO
        assert calculate_sub_index("co", 2000.0) <= 50
        assert calculate_sub_index("co", 8000.0) == pytest.approx(76, abs=3)

    def test_unsupported_pollutant_returns_none(self) -> None:
        assert calculate_sub_index("nh3", 50.0) is None
        assert calculate_sub_index("unknown", 10.0) is None


# =============================================================================
# Sentinel & Missing Value Safety Tests
# =============================================================================

class TestSentinelSafety:
    """Verify that sentinel and invalid values are safely ignored without errors."""

    def test_sentinel_negative_9999(self) -> None:
        assert calculate_sub_index("pm2_5", -9999) is None
        assert calculate_sub_index("no2", -9999) is None

    def test_negative_concentrations(self) -> None:
        assert calculate_sub_index("pm2_5", -5.0) is None
        assert calculate_sub_index("o3", -1.0) is None

    def test_none_and_nan(self) -> None:
        assert calculate_sub_index("pm2_5", None) is None
        assert calculate_sub_index("pm2_5", float("nan")) is None
        assert calculate_sub_index("pm2_5", np.nan) is None


# =============================================================================
# Overall AQI & Dominant Pollutant Tests
# =============================================================================

class TestOverallAQI:
    """Test overall AQI calculation and dominant pollutant determination."""

    def test_dominant_pollutant_pm25(self) -> None:
        pollutants = {
            "pm2_5": 78.5,  # Unhealthy (~163)
            "pm10": 45.0,   # Good (<50)
            "no2": 30.0,    # Good (<50)
            "so2": 10.0,    # Good (<50)
            "co": 500.0,    # Good (<50)
            "o3": 40.0,     # Good (<50)
        }
        overall, dominant, sub_indices = calculate_overall_aqi(pollutants)
        assert dominant == "pm2_5"
        assert overall is not None and overall > 150
        assert sub_indices["pm2_5"] == overall

    def test_dominant_pollutant_o3(self) -> None:
        pollutants = {
            "pm2_5": 8.0,   # Good (44)
            "pm10": 20.0,   # Good (19)
            "o3": 250.0,    # Very Unhealthy (223)
        }
        overall, dominant, sub_indices = calculate_overall_aqi(pollutants)
        assert dominant == "o3"
        assert overall is not None and overall > 200

    def test_overall_aqi_with_sentinels_present(self) -> None:
        pollutants = {
            "pm2_5": 40.0,  # USG (~112)
            "no2": -9999,   # Sentinel missing value
            "co": None,     # Missing
        }
        overall, dominant, sub_indices = calculate_overall_aqi(pollutants)
        assert dominant == "pm2_5"
        assert overall is not None
        assert "no2" not in sub_indices

    def test_empty_pollutants_returns_none(self) -> None:
        assert calculate_overall_aqi({}) == (None, None, {})
        assert calculate_overall_aqi({"pm2_5": -9999}) == (None, None, {})


# =============================================================================
# Category Mapping Tests
# =============================================================================

class TestCategoryMapping:
    """Test AQI category name and color code mapping."""

    def test_category_ranges(self) -> None:
        assert get_aqi_category(25) == ("Good", "#00E400")
        assert get_aqi_category(50) == ("Good", "#00E400")
        assert get_aqi_category(51) == ("Moderate", "#FFFF00")
        assert get_aqi_category(100) == ("Moderate", "#FFFF00")
        assert get_aqi_category(101) == ("Unhealthy for Sensitive Groups", "#FF7E00")
        assert get_aqi_category(150) == ("Unhealthy for Sensitive Groups", "#FF7E00")
        assert get_aqi_category(151) == ("Unhealthy", "#FF0000")
        assert get_aqi_category(200) == ("Unhealthy", "#FF0000")
        assert get_aqi_category(201) == ("Very Unhealthy", "#8F3F97")
        assert get_aqi_category(300) == ("Very Unhealthy", "#8F3F97")
        assert get_aqi_category(301) == ("Hazardous", "#7E0023")
        assert get_aqi_category(500) == ("Hazardous", "#7E0023")

    def test_invalid_and_out_of_bounds_category(self) -> None:
        assert get_aqi_category(None) == ("Unknown", "#808080")
        assert get_aqi_category(float("nan")) == ("Unknown", "#808080")
        assert get_aqi_category(600) == ("Hazardous", "#7E0023")
        assert get_aqi_category(-10) == ("Good", "#00E400")  # Clamped to 0
