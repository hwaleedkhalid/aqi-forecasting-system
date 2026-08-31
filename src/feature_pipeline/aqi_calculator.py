"""Pearls AQI Predictor - US EPA Air Quality Index (AQI) Calculator.

Implements the official United States Environmental Protection Agency (US EPA)
piecewise linear interpolation standard for calculating AQI sub-indices from
raw criteria pollutant concentrations (PM2.5, PM10, O3, NO2, SO2, CO).
"""

from typing import Any
import numpy as np

from src.config import AQI_CATEGORIES

# =============================================================================
# US EPA AQI Breakpoint Definitions (Concentrations in μg/m³)
# Format: (C_low, C_high, I_low, I_high)
# =============================================================================

EPA_BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    # PM2.5 (24-hr avg / hourly proxy, μg/m³)
    "pm2_5": [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 500.4, 401, 500),
    ],
    # PM10 (24-hr avg / hourly proxy, μg/m³)
    "pm10": [
        (0.0, 54.0, 0, 50),
        (55.0, 154.0, 51, 100),
        (155.0, 254.0, 101, 150),
        (255.0, 354.0, 151, 200),
        (355.0, 424.0, 201, 300),
        (425.0, 504.0, 301, 400),
        (505.0, 604.0, 401, 500),
    ],
    # Ozone (O3, 8-hr / 1-hr proxy, μg/m³)
    "o3": [
        (0.0, 107.0, 0, 50),
        (108.0, 137.0, 51, 100),
        (138.0, 166.0, 101, 150),
        (167.0, 205.0, 151, 200),
        (206.0, 392.0, 201, 300),
        (393.0, 793.0, 301, 400),
        (794.0, 989.0, 401, 500),
    ],
    # Nitrogen Dioxide (NO2, 1-hr proxy, μg/m³)
    "no2": [
        (0.0, 100.0, 0, 50),
        (101.0, 188.0, 51, 100),
        (189.0, 677.0, 101, 150),
        (678.0, 1221.0, 151, 200),
        (1222.0, 2350.0, 201, 300),
        (2351.0, 3099.0, 301, 400),
        (3100.0, 3850.0, 401, 500),
    ],
    # Sulfur Dioxide (SO2, 1-hr proxy, μg/m³)
    "so2": [
        (0.0, 92.0, 0, 50),
        (93.0, 196.0, 51, 100),
        (197.0, 485.0, 101, 150),
        (486.0, 797.0, 151, 200),
        (798.0, 1583.0, 201, 300),
        (1584.0, 2108.0, 301, 400),
        (2109.0, 2632.0, 401, 500),
    ],
    # Carbon Monoxide (CO, 8-hr / 1-hr proxy, μg/m³)
    "co": [
        (0.0, 5038.0, 0, 50),
        (5039.0, 10763.0, 51, 100),
        (10764.0, 14200.0, 101, 150),
        (14201.0, 17633.0, 151, 200),
        (17634.0, 34810.0, 201, 300),
        (34811.0, 46258.0, 301, 400),
        (46259.0, 57708.0, 401, 500),
    ],
}


def truncate_concentration(pollutant: str, concentration: float) -> float:
    """Apply EPA concentration truncation rules before breakpoint matching.

    Args:
        pollutant: Pollutant identifier (e.g., 'pm2_5', 'pm10').
        concentration: Raw numeric concentration in μg/m³.

    Returns:
        Truncated concentration value.
    """
    if pollutant == "pm2_5":
        # PM2.5 truncated/rounded to 1 decimal place
        return round(float(concentration), 1)
    elif pollutant in ["pm10", "co", "no2", "so2", "o3"]:
        # Other criteria pollutants rounded to nearest integer or 1 decimal
        return round(float(concentration), 1)
    return float(concentration)


def calculate_sub_index(pollutant: str, concentration: float | None) -> int | None:
    """Calculate EPA AQI sub-index for a specific pollutant concentration.

    Formula:
        I_p = ((I_high - I_low) / (C_high - C_low)) * (C_p - C_low) + I_low

    Args:
        pollutant: Pollutant code ('pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co').
        concentration: Pollutant concentration in μg/m³.

    Returns:
        Integer AQI sub-index in range [0, 500], or None if concentration is invalid/missing.
    """
    if concentration is None or np.isnan(concentration):
        return None

    # Sentinel value check (-9999 or any negative value)
    if concentration < 0 or concentration == -9999:
        return None

    breakpoints = EPA_BREAKPOINTS.get(pollutant.lower())
    if not breakpoints:
        return None

    c_p = truncate_concentration(pollutant.lower(), concentration)

    # Below lowest breakpoint
    if c_p <= breakpoints[0][0]:
        return breakpoints[0][2]

    # Matching within standard breakpoint range
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c_p <= c_high:
            sub_index = ((i_high - i_low) / (c_high - c_low)) * (c_p - c_low) + i_low
            return int(round(sub_index))

    # Above highest defined breakpoint (> 500)
    return 500


def calculate_overall_aqi(
    pollutants: dict[str, Any]
) -> tuple[int | None, str | None, dict[str, int]]:
    """Calculate overall US EPA AQI, dominant pollutant, and individual sub-indices.

    The overall AQI is the maximum of all valid criteria pollutant sub-indices:
        Overall AQI = max(I_PM2.5, I_PM10, I_O3, I_NO2, I_SO2, I_CO)

    Args:
        pollutants: Dictionary mapping pollutant names to concentrations in μg/m³.

    Returns:
        Tuple of (overall_aqi, dominant_pollutant_name, dictionary_of_sub_indices).
        Returns (None, None, {}) if no valid criteria pollutant concentrations are present.
    """
    sub_indices: dict[str, int] = {}

    for pol in ["pm2_5", "pm10", "o3", "no2", "so2", "co"]:
        val = pollutants.get(pol)
        if val is not None:
            sub_idx = calculate_sub_index(pol, val)
            if sub_idx is not None:
                sub_indices[pol] = sub_idx

    if not sub_indices:
        return None, None, {}

    # Dominant pollutant is the one with highest sub-index
    dominant_pollutant = max(sub_indices, key=lambda k: sub_indices[k])
    overall_aqi = int(min(500, max(0, sub_indices[dominant_pollutant])))

    return overall_aqi, dominant_pollutant, sub_indices


def get_aqi_category(aqi: int | None) -> tuple[str, str]:
    """Map an AQI integer value to its EPA Category name and Hex color code.

    Args:
        aqi: Integer AQI value in range [0, 500].

    Returns:
        Tuple of (Category Name, Hex Color).
    """
    if aqi is None or np.isnan(aqi):
        return "Unknown", "#808080"

    aqi_clamped = int(min(500, max(0, round(aqi))))

    for cat_name, cat_info in AQI_CATEGORIES.items():
        if cat_info["min"] <= aqi_clamped <= cat_info["max"]:
            return cat_name, cat_info["color"]

    # Fallback for unexpected dictionary structure
    return "Unknown", "#808080"  # pragma: no cover
