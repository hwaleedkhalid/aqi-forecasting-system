"""Pearls AQI Predictor - Phase 11.5: Winter/Smog Candidate Feature Families.

Constructs four independent feature families hypothesized to contain predictive
information associated with winter/smog atmospheric conditions in Lahore.

All features are:
  - Derived exclusively from variables available at timestamp t (backward-looking only).
  - Computable without external APIs (Open-Meteo + pollutant data only).
  - Described as proxies / calendar-based indicators, not as direct physical measurements.

Feature Families:
  1. Thermal/Inversion-Risk Proxy  — diurnal temperature range + composite proxy
  2. Fog/Mist Indicator            — humidity + temperature + wind threshold flags
  3. Stagnation Enhancement        — rolling low-wind counts + pressure stability
  4. Seasonal-Emission Proxy       — crop-burning calendar flag + interaction term

IMPORTANT: These features are hypotheses under controlled ablation evaluation.
  - inversion_risk_proxy: A meteorological proxy associated with inversion-prone
    conditions. It does NOT directly measure temperature inversions (which require
    vertical atmospheric profile data not available here).
  - crop_burning_season: A calendar-based seasonal emission proxy corresponding to
    Punjab wheat straw burning (Oct-15 to Nov-30). It is NOT derived from actual
    fire-count observations (e.g. FIRMS/MODIS), which are unavailable in this dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logger import logger


# =============================================================================
# Constants
# =============================================================================

# Punjab wheat straw burning season (calendar-based proxy, not fire-count derived)
CROP_BURNING_START_MONTH = 10
CROP_BURNING_START_DAY = 15
CROP_BURNING_END_MONTH = 11
CROP_BURNING_END_DAY = 30

# Fog/mist proxy thresholds (physically motivated but not derived from visibility data)
FOG_HUMIDITY_THRESHOLD = 85.0    # %
FOG_TEMPERATURE_THRESHOLD = 12.0  # °C
FOG_WIND_THRESHOLD = 2.0          # m/s

# Wind stagnation threshold
WIND_STAGNATION_THRESHOLD = 1.5  # m/s — near-calm

FAMILY_NAMES = {
    "family1_inversion": [
        "diurnal_temp_range_24h",
        "inversion_risk_proxy",
        "temp_drop_6h",
    ],
    "family2_fog": [
        "fog_proxy",
        "fog_hours_rolling_24h",
    ],
    "family3_stagnation": [
        "wind_stagnation_hours_12h",
        "wind_stagnation_hours_24h",
        "pressure_stability_24h",
    ],
    "family4_emission": [
        "crop_burning_season",
        "winter_emission_intensity",
    ],
}

ALL_WINTER_FEATURE_COLS = [col for cols in FAMILY_NAMES.values() for col in cols]


# =============================================================================
# Family 1: Thermal / Inversion-Risk Proxy
# =============================================================================

def add_thermal_inversion_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Family 1: Thermal / Inversion-Risk Proxy features.

    These features capture meteorological patterns associated with inversion-prone
    atmospheric conditions in Lahore winters (low diurnal temperature range, high
    humidity, weak winds). They are proxies — NOT direct measurements of inversion
    layers (which require vertical profile data unavailable here).

    Features:
        diurnal_temp_range_24h: Backward-looking 24h rolling (max - min) of temperature.
            Low values indicate suppressed convective mixing (inversion-prone).
        inversion_risk_proxy: Normalized composite score [0, 1]. Higher = more
            inversion-prone conditions. Derived from low DTR + high RH + low wind.
        temp_drop_6h: Temperature change over the past 6 hours (negative = cooling).
            Rapid nighttime cooling increases inversion risk.

    Args:
        df: Feature dataframe with 'temperature_2m', 'relative_humidity_2m',
            'wind_speed_10m' columns and datetime_utc index or column.

    Returns:
        DataFrame with 3 new Family 1 columns added (in-place copy).
    """
    df = df.copy()

    # Diurnal Temperature Range (backward-looking, no future info)
    df["diurnal_temp_range_24h"] = (
        df["temperature_2m"].rolling(window=24, min_periods=12).max()
        - df["temperature_2m"].rolling(window=24, min_periods=12).min()
    )

    # Temperature drop over past 6 hours (negative = cooling → inversion risk ↑)
    df["temp_drop_6h"] = df["temperature_2m"] - df["temperature_2m"].shift(6)

    # Inversion-risk proxy: composite normalized score
    # Each component normalized to [0, 1], then averaged
    dtr = df["diurnal_temp_range_24h"].clip(lower=0.0)
    dtr_max = float(max(dtr.quantile(0.99), 1e-6))
    low_dtr_score = 1.0 - (dtr / dtr_max).clip(0.0, 1.0)  # 1 = low DTR = inversion-prone

    rh = df["relative_humidity_2m"].clip(0.0, 100.0)
    high_rh_score = (rh / 100.0)  # 1 = saturated

    ws = df["wind_speed_10m"].clip(lower=0.0)
    ws_max = float(max(ws.quantile(0.99), 1e-6))
    low_wind_score = 1.0 - (ws / ws_max).clip(0.0, 1.0)  # 1 = calm

    df["inversion_risk_proxy"] = (low_dtr_score + high_rh_score + low_wind_score) / 3.0

    logger.debug(
        f"Family 1 (thermal/inversion-risk proxy): "
        f"diurnal_temp_range_24h mean={df['diurnal_temp_range_24h'].mean():.2f}, "
        f"inversion_risk_proxy mean={df['inversion_risk_proxy'].mean():.3f}"
    )
    return df


# =============================================================================
# Family 2: Fog/Mist Indicator
# =============================================================================

def add_fog_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Family 2: Fog/Mist Indicator features.

    Fog conditions trap pollutants near the surface. The threshold-based binary flag
    is a proxy derived from surface humidity, temperature, and wind — NOT from actual
    visibility observations (which are unavailable in this dataset).

    Features:
        fog_proxy: Binary flag (1 = fog-prone conditions present).
            Condition: RH > 85% AND temp < 12°C AND wind < 2.0 m/s.
        fog_hours_rolling_24h: Count of fog-proxy hours in the past 24 hours.
            Captures persistent fog episodes.

    Args:
        df: Feature dataframe with 'relative_humidity_2m', 'temperature_2m',
            'wind_speed_10m' columns.

    Returns:
        DataFrame with 2 new Family 2 columns added.
    """
    df = df.copy()

    fog_condition = (
        (df["relative_humidity_2m"] > FOG_HUMIDITY_THRESHOLD)
        & (df["temperature_2m"] < FOG_TEMPERATURE_THRESHOLD)
        & (df["wind_speed_10m"] < FOG_WIND_THRESHOLD)
    )
    df["fog_proxy"] = fog_condition.astype(np.float32)

    df["fog_hours_rolling_24h"] = (
        df["fog_proxy"].rolling(window=24, min_periods=1).sum().astype(np.float32)
    )

    logger.debug(
        f"Family 2 (fog): fog_proxy fraction={df['fog_proxy'].mean():.4f}, "
        f"fog_hours_rolling_24h mean={df['fog_hours_rolling_24h'].mean():.2f}"
    )
    return df


# =============================================================================
# Family 3: Stagnation Enhancement
# =============================================================================

def add_stagnation_enhancement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Family 3: Stagnation Enhancement features.

    These extend the existing `stagnation_index` (PM2.5 / wind) with rolling
    near-calm wind duration and pressure stability metrics. The question under
    ablation is whether these add signal beyond what `stagnation_index` already
    captures.

    Features:
        wind_stagnation_hours_12h: Count of hours in past 12h with wind < 1.5 m/s.
        wind_stagnation_hours_24h: Count of hours in past 24h with wind < 1.5 m/s.
        pressure_stability_24h: Rolling 24h std of surface pressure (low = stable,
            no frontal activity).

    Args:
        df: Feature dataframe with 'wind_speed_10m', 'surface_pressure' columns.

    Returns:
        DataFrame with 3 new Family 3 columns added.
    """
    df = df.copy()

    near_calm = (df["wind_speed_10m"] < WIND_STAGNATION_THRESHOLD).astype(np.float32)

    df["wind_stagnation_hours_12h"] = (
        near_calm.rolling(window=12, min_periods=1).sum().astype(np.float32)
    )
    df["wind_stagnation_hours_24h"] = (
        near_calm.rolling(window=24, min_periods=1).sum().astype(np.float32)
    )

    # Low pressure std = synoptic stability (anticyclonic = stagnation-prone)
    df["pressure_stability_24h"] = (
        df["surface_pressure"]
        .rolling(window=24, min_periods=12)
        .std()
        .fillna(0.0)
        .astype(np.float32)
    )

    logger.debug(
        f"Family 3 (stagnation): wind_stagnation_hours_24h mean="
        f"{df['wind_stagnation_hours_24h'].mean():.2f}"
    )
    return df


# =============================================================================
# Family 4: Seasonal-Emission Proxy
# =============================================================================

def add_seasonal_emission_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Family 4: Seasonal-Emission Proxy features.

    crop_burning_season is a CALENDAR-BASED proxy for the Punjab wheat straw
    burning season (Oct-15 to Nov-30). It is NOT derived from actual fire-count
    observations (FIRMS/MODIS data is unavailable). It encodes the hypothesis
    that emission intensity from agricultural burning peaks during this window.

    Features:
        crop_burning_season: Binary flag (1 = within Oct-15 to Nov-30 window).
        winter_emission_intensity: crop_burning_season × pm_ratio (interaction term
            combining the seasonal window with the PM2.5/PM10 ratio as a combustion
            source signature).

    Args:
        df: Feature dataframe with datetime_utc column and 'pm_ratio' column.
            If 'pm_ratio' is absent, winter_emission_intensity is set to NaN.

    Returns:
        DataFrame with 2 new Family 4 columns added.
    """
    df = df.copy()

    dt = pd.to_datetime(df["datetime_utc"], utc=True)
    month = dt.dt.month
    day = dt.dt.day

    crop_burning = (
        ((month == CROP_BURNING_START_MONTH) & (day >= CROP_BURNING_START_DAY))
        | (month == CROP_BURNING_END_MONTH)
    )
    df["crop_burning_season"] = crop_burning.astype(np.float32)

    if "pm_ratio" in df.columns:
        df["winter_emission_intensity"] = (
            df["crop_burning_season"] * df["pm_ratio"]
        ).astype(np.float32)
    else:
        df["winter_emission_intensity"] = np.nan

    logger.debug(
        f"Family 4 (emission proxy): crop_burning_season fraction="
        f"{df['crop_burning_season'].mean():.4f}"
    )
    return df


# =============================================================================
# Unified Enrichment Entry Point
# =============================================================================

def enrich_with_winter_features(
    df: pd.DataFrame,
    families: list[str] | None = None,
) -> pd.DataFrame:
    """Add selected winter/smog candidate feature families to a feature dataframe.

    Args:
        df: Feature dataframe. Must contain 'datetime_utc', 'temperature_2m',
            'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure', 'pm_ratio'.
        families: List of family names to add. If None, adds all four families.
            Options: 'family1_inversion', 'family2_fog', 'family3_stagnation',
                     'family4_emission'.

    Returns:
        Enriched dataframe with new feature columns appended.
    """
    if families is None:
        families = list(FAMILY_NAMES.keys())

    enrichers = {
        "family1_inversion": add_thermal_inversion_proxy,
        "family2_fog": add_fog_indicators,
        "family3_stagnation": add_stagnation_enhancement,
        "family4_emission": add_seasonal_emission_proxy,
    }

    for family in families:
        if family not in enrichers:
            raise ValueError(f"Unknown family: {family!r}. Valid: {list(enrichers.keys())}")
        df = enrichers[family](df)
        new_cols = FAMILY_NAMES[family]
        n_nan = df[new_cols].isnull().sum().sum()
        logger.info(f"Added {family}: {len(new_cols)} features, {n_nan} NaN values.")

    return df
