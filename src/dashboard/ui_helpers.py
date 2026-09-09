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
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy for Sensitive Groups": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
}

_CATEGORY_TEXT_COLORS: dict[str, str] = {
    "Good": "#1F2937",
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


def category_color(category: str, fallback: str = "#9CA3AF") -> str:
    """Return EPA hex color for a category string."""
    return _CATEGORY_COLORS.get(category, fallback)


def category_text_color(category: str) -> str:
    """Return readable text color (light/dark) for overlay on category background."""
    return _CATEGORY_TEXT_COLORS.get(category, "#FFFFFF")


def category_icon(category: str) -> str:
    """Return emoji icon for a category."""
    return _CATEGORY_ICONS.get(category, "🌫️")
