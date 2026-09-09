"""Pearls AQI Predictor - Dashboard UI Utility Helpers.

Lightweight formatting and safe-value functions used across components.
No Streamlit imports; purely functional, easily unit-tested.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


# ── Value safety ──────────────────────────────────────────────────────────────

def safe_val(
    v: Any,
    unit: str = "",
    decimals: int = 1,
    fallback: str = "—",
) -> str:
    """Format a numeric value with unit, returning fallback for None/nan/missing."""
    if v is None:
        return fallback
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return fallback
        formatted = f"{f:.{decimals}f}"
        return f"{formatted} {unit}".strip() if unit else formatted
    except (TypeError, ValueError):
        return fallback


# ── Time formatting ───────────────────────────────────────────────────────────

def format_relative_age(age_hours: float | None) -> str:
    """Convert age in hours to a human-readable relative string.

    Examples:
        0.05  → "just now"
        0.4   → "24 min ago"
        1.0   → "1.0h ago"
        4.1   → "4.1h ago"
        25.0  → "25.0h ago"
    """
    if age_hours is None:
        return "unknown time"
    try:
        h = float(age_hours)
    except (TypeError, ValueError):
        return "unknown time"

    if h < 0.017:  # < ~1 min
        return "just now"
    if h < 1.0:
        minutes = round(h * 60)
        return f"{minutes} min ago"
    return f"{h:.1f}h ago"


def format_stale_age(age_hours: float | None) -> str:
    """Format a stale/historical age notice."""
    if age_hours is None:
        return "historical observation"
    try:
        h = float(age_hours)
    except (TypeError, ValueError):
        return "historical observation"
    return f"historical · {h:.1f}h old"


def format_timestamp(iso_str: str | None, fallback: str = "—") -> str:
    """Return a compact human-readable date-time from ISO 8601 string.

    E.g. "2026-09-09T14:32:00+00:00" → "2026-09-09 14:32 UTC"
    """
    if not iso_str or iso_str == "Unknown":
        return fallback
    try:
        dt = datetime.fromisoformat(iso_str)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%d %H:%M UTC")
    except (ValueError, TypeError):
        return str(iso_str)


def format_horizon_label(horizon: int | None) -> str:
    """Format a horizon integer as '+Nh'."""
    if horizon is None:
        return "—"
    try:
        return f"+{int(horizon)}h"
    except (TypeError, ValueError):
        return "—"


# ── Source mode helpers ───────────────────────────────────────────────────────

def source_display_name(
    feature_source: str,
    fallback_active: bool,
    is_stale: bool,
) -> str:
    """Return user-friendly data source label."""
    if fallback_active or feature_source in ("bootstrap", "local_fallback"):
        return "Cached data"
    if feature_source == "hopsworks":
        return "Hopsworks"
    if "REST API" in (feature_source or ""):
        return "Live API"
    return "Live"


def source_badge_class(feature_source: str, fallback_active: bool, is_stale: bool) -> str:
    """Return CSS class for the source badge."""
    if is_stale:
        return "prl-source-badge prl-source-badge-stale"
    if fallback_active or feature_source in ("bootstrap", "local_fallback"):
        return "prl-source-badge prl-source-badge-fallback"
    return "prl-source-badge"


# ── AQI category color lookup ─────────────────────────────────────────────────

# EPA AQI standard category colors
_CATEGORY_COLORS: dict[str, str] = {
    "Good": "#22C55E",
    "Moderate": "#EAB308",
    "Unhealthy for Sensitive Groups": "#F97316",
    "Unhealthy": "#EF4444",
    "Very Unhealthy": "#8B5CF6",
    "Hazardous": "#7F1D1D",
}

_CATEGORY_BG_COLORS: dict[str, str] = {
    "Good": "#F0FDF4",
    "Moderate": "#FEFCE8",
    "Unhealthy for Sensitive Groups": "#FFF7ED",
    "Unhealthy": "#FEF2F2",
    "Very Unhealthy": "#FAF5FF",
    "Hazardous": "#FFF1F2",
}

_CATEGORY_BORDER_COLORS: dict[str, str] = {
    "Good": "#BBF7D0",
    "Moderate": "#FEF08A",
    "Unhealthy for Sensitive Groups": "#FED7AA",
    "Unhealthy": "#FECACA",
    "Very Unhealthy": "#E9D5FF",
    "Hazardous": "#FFE4E6",
}

_CATEGORY_TEXT_COLORS: dict[str, str] = {
    "Good": "#15803D",
    "Moderate": "#854D0E",
    "Unhealthy for Sensitive Groups": "#C2410C",
    "Unhealthy": "#B91C1C",
    "Very Unhealthy": "#6D28D9",
    "Hazardous": "#9F1239",
}

_CATEGORY_SOLID_TEXT_COLORS: dict[str, str] = {
    "Good": "#FFFFFF",
    "Moderate": "#1F2937",
    "Unhealthy for Sensitive Groups": "#FFFFFF",
    "Unhealthy": "#FFFFFF",
    "Very Unhealthy": "#FFFFFF",
    "Hazardous": "#FFFFFF",
}

_CATEGORY_ICONS: dict[str, str] = {
    "Good": "🌿",
    "Moderate": "😐",
    "Unhealthy for Sensitive Groups": "⚠️",
    "Unhealthy": "🔴",
    "Very Unhealthy": "🟣",
    "Hazardous": "☠️",
}

_CATEGORY_CARD_CLASSES: dict[str, str] = {
    "Good": "prl-card prl-card-good",
    "Moderate": "prl-card prl-card-moderate",
    "Unhealthy for Sensitive Groups": "prl-card prl-card-usg",
    "Unhealthy": "prl-card prl-card-unhealthy",
    "Very Unhealthy": "prl-card prl-card-very-unhealthy",
    "Hazardous": "prl-card prl-card-hazardous",
}


def category_color(category: str, fallback: str = "#9CA3AF") -> str:
    """Return EPA hex accent color for a category string."""
    return _CATEGORY_COLORS.get(category, fallback)


def category_bg_color(category: str, fallback: str = "#FFFFFF") -> str:
    """Return soft background tint hex color for a category."""
    return _CATEGORY_BG_COLORS.get(category, fallback)


def category_border_color(category: str, fallback: str = "#E5E7EB") -> str:
    """Return border hex color for a category."""
    return _CATEGORY_BORDER_COLORS.get(category, fallback)


def category_text_color(category: str) -> str:
    """Return text color matching category palette."""
    return _CATEGORY_TEXT_COLORS.get(category, "#1F2937")


def category_solid_text_color(category: str) -> str:
    """Return readable text color (light/dark) for solid accent pill background."""
    return _CATEGORY_SOLID_TEXT_COLORS.get(category, "#FFFFFF")


def category_icon(category: str) -> str:
    """Return emoji icon for a category."""
    return _CATEGORY_ICONS.get(category, "🌫️")


def category_card_class(category: str) -> str:
    """Return CSS class name for a category-tinted card."""
    return _CATEGORY_CARD_CLASSES.get(category, "prl-card")


# ── Feature explainability helpers ────────────────────────────────────────────

def humanize_feature_name(feature_name: str) -> str:
    """Translate raw ML feature name into a clear human-readable description."""
    if not feature_name:
        return "Unknown Factor"

    mapping = {
        "pm2_5": "PM2.5 concentration",
        "pm10": "PM10 concentration",
        "no2": "NO₂ concentration",
        "so2": "SO₂ concentration",
        "co": "CO concentration",
        "o3": "Ozone (O₃) concentration",
        "temperature_2m": "Air temperature",
        "relative_humidity_2m": "Relative humidity",
        "wind_speed_10m": "Wind speed",
        "surface_pressure": "Surface pressure",
        "pm2_5_lag_1h": "Recent PM2.5 level",
        "pm10_lag_1h": "Recent PM10 level",
        "no2_lag_1h": "Recent NO₂ level",
        "so2_lag_1h": "Recent SO₂ level",
        "co_lag_1h": "Recent CO level",
        "o3_lag_1h": "Recent Ozone (O₃) level",
        "epa_aqi_lag_1h": "Recent AQI level",
        "epa_aqi_rolling_mean_6h": "6-hour AQI trend",
        "epa_aqi_rolling_mean_12h": "12-hour AQI trend",
        "epa_aqi_rolling_mean_24h": "24-hour AQI trend",
        "pm2_5_rolling_mean_6h": "6-hour PM2.5 trend",
        "pm2_5_rolling_mean_24h": "24-hour PM2.5 trend",
        "pm10_rolling_mean_6h": "6-hour PM10 trend",
        "pm10_rolling_mean_24h": "24-hour PM10 trend",
        "temperature_2m_lag_1h": "Recent temperature",
        "relative_humidity_2m_lag_1h": "Recent humidity",
        "wind_speed_10m_lag_1h": "Recent wind speed",
        "surface_pressure_lag_1h": "Recent atmospheric pressure",
        "hour_of_day_sin": "Daily diurnal cycle",
        "hour_of_day_cos": "Daily diurnal cycle",
        "day_of_week_sin": "Day of week pattern",
        "day_of_week_cos": "Day of week pattern",
        "month_sin": "Seasonal pattern",
        "month_cos": "Seasonal pattern",
    }
    if feature_name in mapping:
        return mapping[feature_name]

    # Specific common patterns
    feat_lower = feature_name.lower()
    if "temperature" in feat_lower and "rolling" in feat_lower:
        return "Recent temperature trend"
    if "humidity" in feat_lower and "rolling" in feat_lower:
        return "Recent humidity trend"
    if "wind" in feat_lower and "rolling" in feat_lower:
        return "Recent wind speed trend"
    if "pm10" in feat_lower:
        return "PM10 level"
    if "pm2_5" in feat_lower or "pm25" in feat_lower:
        return "PM2.5 level"

    # Generic cleanup
    clean = feature_name.replace("_lag_1h", " (recent)").replace("_rolling_mean_", " (rolling avg ")
    clean = clean.replace("_2m", "").replace("_10m", "").replace("_", " ").strip()
    return clean.capitalize()



def format_shap_narrative(
    top_features: list[dict[str, Any]],
    default_horizon: int = 24,
) -> list[dict[str, Any]]:
    """Derive signed, non-causal attribution narratives from top SHAP features."""
    narratives = []
    for item in top_features:
        feat = item.get("feature", "")
        shap_val = float(item.get("shap_value", 0.0))
        raw_val = item.get("raw_value")

        human_name = humanize_feature_name(feat)
        is_upward = shap_val > 0
        direction_arrow = "↑" if is_upward else "↓"
        direction_word = "upward" if is_upward else "downward"

        abs_val = abs(shap_val)
        if abs_val >= 15.0:
            strength = "Strong"
            badge_class = "prl-shap-badge-strong"
        elif abs_val >= 5.0:
            strength = "Moderate"
            badge_class = "prl-shap-badge-mod"
        else:
            strength = "Minor"
            badge_class = "prl-shap-badge-minor"

        sign_str = f"+{abs_val:.1f}" if is_upward else f"-{abs_val:.1f}"
        statement = (
            f"{direction_arrow} {human_name} contributed {direction_word} pressure "
            f"({sign_str} AQI points) relative to baseline."
        )

        narratives.append({
            "feature": feat,
            "human_name": human_name,
            "shap_value": shap_val,
            "raw_value": raw_val,
            "is_upward": is_upward,
            "arrow": direction_arrow,
            "direction": direction_word,
            "strength": strength,
            "badge_class": badge_class,
            "statement": statement,
        })
    return narratives

