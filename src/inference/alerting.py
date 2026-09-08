"""Pearls AQI Predictor - Centralized AQI Alert & Severity Evaluator.

Authoritative source of truth for AQI alert classification, severity ranking,
and multi-horizon forecast alert evaluation across Flask REST API and Streamlit dashboard.

Strictly follows the project architecture:
AQI numeric value -> get_aqi_category() -> category -> category-to-alert mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.feature_pipeline.aqi_calculator import get_aqi_category

# Official US EPA Health Advisories by Category
HEALTH_ADVISORIES: dict[str, str] = {
    "Good": "Air quality is satisfactory and air pollution poses little or no risk.",
    "Moderate": "Air quality is acceptable; unusually sensitive individuals should consider reducing prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups (children, elderly, respiratory/heart conditions) may experience health effects. General public is less likely to be affected.",
    "Unhealthy": "Some members of the general public may experience health effects; sensitive groups should avoid prolonged outdoor exertion.",
    "Very Unhealthy": "Health alert: The risk of health effects is increased for everyone. Avoid strenuous outdoor activities.",
    "Hazardous": "Health warning of emergency conditions: Everyone is more likely to be affected. Remain indoors and keep activity levels low.",
}

# =============================================================================
# Alert Severity Metadata Contract
# =============================================================================

CATEGORY_TO_ALERT_CONFIG: dict[str, dict[str, Any]] = {
    "Good": {
        "level": "none",
        "severity_rank": 0,
        "active": False,
        "threshold": None,
        "title": "Good Air Quality",
    },
    "Moderate": {
        "level": "none",
        "severity_rank": 1,
        "active": False,
        "threshold": None,
        "title": "Moderate Air Quality",
    },
    "Unhealthy for Sensitive Groups": {
        "level": "advisory",
        "severity_rank": 2,
        "active": True,
        "threshold": 101.0,
        "title": "Air Quality Advisory",
    },
    "Unhealthy": {
        "level": "warning",
        "severity_rank": 3,
        "active": True,
        "threshold": 151.0,
        "title": "Unhealthy Air Quality Warning",
    },
    "Very Unhealthy": {
        "level": "severe",
        "severity_rank": 4,
        "active": True,
        "threshold": 201.0,
        "title": "Very Unhealthy Air Quality Alert",
    },
    "Hazardous": {
        "level": "hazardous",
        "severity_rank": 5,
        "active": True,
        "threshold": 301.0,
        "title": "Hazardous Air Quality Emergency",
    },
}


@dataclass(frozen=True)
class AQIAlert:
    """Normalized alert metadata structure for an individual AQI observation or point."""

    active: bool
    level: str  # "none" | "advisory" | "warning" | "severe" | "hazardous"
    severity_rank: int  # 0..5
    category: str
    color: str
    aqi: float
    threshold: float | None
    title: str
    message: str
    data_is_stale: bool = False
    feature_source: str = "unknown"
    fallback_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ForecastAlertSummary:
    """Multi-horizon 72-hour forecast alert evaluation summary."""

    active: bool
    highest_level: str  # "none" | "advisory" | "warning" | "severe" | "hazardous"
    highest_severity_rank: int  # 0..5
    peak_aqi: float
    peak_horizon: int
    peak_timestamp: str
    peak_category: str
    first_advisory_horizon: int | None
    first_advisory_timestamp: str | None
    first_unhealthy_horizon: int | None
    first_unhealthy_timestamp: str | None
    first_very_unhealthy_horizon: int | None
    first_very_unhealthy_timestamp: str | None
    first_hazardous_horizon: int | None
    first_hazardous_timestamp: str | None
    advisory_horizon_count: int
    unhealthy_horizon_count: int
    very_unhealthy_horizon_count: int
    hazardous_horizon_count: int
    severe_or_higher_horizon_count: int
    title: str
    message: str
    data_is_stale: bool = False
    fallback_active: bool = False
    feature_source: str = "unknown"
    upper_interval_crosses_hazardous: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to serializable dictionary."""
        return asdict(self)


def classify_category_alert(category: str) -> dict[str, Any]:
    """Map an EPA category name to normalized alert severity properties.

    Args:
        category: EPA Category name (e.g. 'Unhealthy', 'Hazardous').

    Returns:
        Dictionary containing level, severity_rank, active, threshold, and title.
    """
    config = CATEGORY_TO_ALERT_CONFIG.get(category)
    if config:
        return config

    # Fallback for unexpected or Unknown categories
    return {
        "level": "none",
        "severity_rank": 0,
        "active": False,
        "threshold": None,
        "title": f"{category} Air Quality",
    }


def evaluate_current_alert(
    aqi: float,
    data_is_stale: bool = False,
    feature_source: str = "unknown",
    fallback_active: bool = False,
) -> AQIAlert:
    """Evaluate current AQI observation using the authoritative category mapping path.

    Flow:
        numeric AQI -> get_aqi_category() -> category -> classify_category_alert().

    Args:
        aqi: Numeric current AQI value.
        data_is_stale: Whether underlying observation is older than 3 hours.
        feature_source: Name of feature provider ('hopsworks', 'bootstrap', etc.).
        fallback_active: Whether bootstrap fallback is currently active.

    Returns:
        AQIAlert instance.
    """
    category, color = get_aqi_category(aqi)
    config = classify_category_alert(category)
    message = HEALTH_ADVISORIES.get(
        category,
        "Monitor local conditions and adjust outdoor activities as needed.",
    )

    return AQIAlert(
        active=config["active"],
        level=config["level"],
        severity_rank=config["severity_rank"],
        category=category,
        color=color,
        aqi=round(float(aqi), 1),
        threshold=config["threshold"],
        title=config["title"],
        message=message,
        data_is_stale=data_is_stale,
        feature_source=feature_source,
        fallback_active=fallback_active,
    )


def evaluate_forecast_alerts(
    forecasts: list[dict[str, Any]],
    data_is_stale: bool = False,
    fallback_active: bool = False,
    feature_source: str = "unknown",
) -> ForecastAlertSummary:
    """Scan all 72 forecast horizon points and construct an authoritative alert summary.

    Derives alert severity strictly from each forecast point's normalized category/alert
    metadata to guarantee 100% consistency with displayed point values.

    Args:
        forecasts: List of horizon forecast dictionaries matching forecast contract.
        data_is_stale: Provenance staleness indicator.
        fallback_active: Provenance bootstrap indicator.
        feature_source: Provenance feature source name.

    Returns:
        ForecastAlertSummary instance.
    """
    if not forecasts:
        return ForecastAlertSummary(
            active=False,
            highest_level="none",
            highest_severity_rank=0,
            peak_aqi=0.0,
            peak_horizon=0,
            peak_timestamp="",
            peak_category="Unknown",
            first_advisory_horizon=None,
            first_advisory_timestamp=None,
            first_unhealthy_horizon=None,
            first_unhealthy_timestamp=None,
            first_very_unhealthy_horizon=None,
            first_very_unhealthy_timestamp=None,
            first_hazardous_horizon=None,
            first_hazardous_timestamp=None,
            advisory_horizon_count=0,
            unhealthy_horizon_count=0,
            very_unhealthy_horizon_count=0,
            hazardous_horizon_count=0,
            severe_or_higher_horizon_count=0,
            title="No Forecast Available",
            message="No forecast data is currently available.",
            data_is_stale=data_is_stale,
            fallback_active=fallback_active,
            feature_source=feature_source,
            upper_interval_crosses_hazardous=False,
        )

    # 1. Identify peak AQI point across all 72 horizons
    peak_point = max(forecasts, key=lambda p: float(p.get("aqi", 0.0)))
    peak_aqi = round(float(peak_point.get("aqi", 0.0)), 1)
    peak_horizon = int(peak_point.get("horizon", 1))
    peak_timestamp = str(peak_point.get("forecast_time", ""))
    peak_category = str(peak_point.get("category", "Unknown"))

    # 2. Sequential scanning for threshold crossings based on point category/alert metadata
    first_advisory_h: int | None = None
    first_advisory_t: str | None = None
    first_unhealthy_h: int | None = None
    first_unhealthy_t: str | None = None
    first_very_unhealthy_h: int | None = None
    first_very_unhealthy_t: str | None = None
    first_hazardous_h: int | None = None
    first_hazardous_t: str | None = None

    advisory_count = 0
    unhealthy_count = 0
    very_unhealthy_count = 0
    hazardous_count = 0
    max_rank = 0

    upper_crosses_haz = False

    for pt in forecasts:
        # Resolve category and rank directly from point category
        cat = pt.get("category")
        if cat:
            cfg = classify_category_alert(cat)
            rank = cfg["severity_rank"]
        else:
            rank = pt.get("severity_rank", 0)

        if rank > max_rank:
            max_rank = rank

        h = int(pt.get("horizon", 0))
        t = str(pt.get("forecast_time", ""))

        # Check empirical error upper bound
        if float(pt.get("error_upper", 0.0)) >= 301.0:
            upper_crosses_haz = True

        # Threshold crossings: first horizon entering or exceeding level
        if rank >= 2 and first_advisory_h is None:
            first_advisory_h = h
            first_advisory_t = t
        if rank >= 3 and first_unhealthy_h is None:
            first_unhealthy_h = h
            first_unhealthy_t = t
        if rank >= 4 and first_very_unhealthy_h is None:
            first_very_unhealthy_h = h
            first_very_unhealthy_t = t
        if rank >= 5 and first_hazardous_h is None:
            first_hazardous_h = h
            first_hazardous_t = t

        # Category-specific horizon counts
        if rank == 2:
            advisory_count += 1
        elif rank == 3:
            unhealthy_count += 1
        elif rank == 4:
            very_unhealthy_count += 1
        elif rank >= 5:
            hazardous_count += 1

    severe_or_higher_count = very_unhealthy_count + hazardous_count
    active = bool(max_rank >= 2)

    rank_to_level = {
        0: "none",
        1: "none",
        2: "advisory",
        3: "warning",
        4: "severe",
        5: "hazardous",
    }
    highest_level = rank_to_level.get(max_rank, "none")

    # Construct scientifically grounded, uncertainty-aware title and message
    if highest_level == "hazardous":
        title = "Hazardous AQI Forecast"
        first_str = f"First enters Hazardous range at +{first_hazardous_h}h." if first_hazardous_h else ""
        message = (
            f"The model forecasts Hazardous air quality conditions within the next 72 hours "
            f"(Peak AQI {peak_aqi:.1f} at +{peak_horizon}h). {first_str} "
            f"A total of {hazardous_count} forecast hours are in the Hazardous range."
        )
    elif highest_level == "severe":
        title = "Very Unhealthy AQI Forecast"
        first_str = f"First enters Very Unhealthy range at +{first_very_unhealthy_h}h." if first_very_unhealthy_h else ""
        message = (
            f"The model forecasts Very Unhealthy air quality conditions within the next 72 hours "
            f"(Peak AQI {peak_aqi:.1f} at +{peak_horizon}h). {first_str} "
            f"A total of {very_unhealthy_count} forecast hours are in the Very Unhealthy range."
        )
    elif highest_level == "warning":
        title = "Unhealthy AQI Forecast"
        first_str = f"First enters Unhealthy range at +{first_unhealthy_h}h." if first_unhealthy_h else ""
        message = (
            f"The model forecasts Unhealthy air quality conditions within the next 72 hours "
            f"(Peak AQI {peak_aqi:.1f} at +{peak_horizon}h). {first_str} "
            f"A total of {unhealthy_count} forecast hours are in the Unhealthy range."
        )
    elif highest_level == "advisory":
        title = "Air Quality Advisory Forecast"
        message = (
            f"The model forecasts air quality Unhealthy for Sensitive Groups within the next 72 hours "
            f"(Peak AQI {peak_aqi:.1f} at +{peak_horizon}h)."
        )
    else:
        title = "Normal Air Quality Forecast"
        message = (
            f"The model forecasts Good to Moderate air quality across all 72 forecast horizons "
            f"(Peak AQI {peak_aqi:.1f} at +{peak_horizon}h)."
        )

    return ForecastAlertSummary(
        active=active,
        highest_level=highest_level,
        highest_severity_rank=max_rank,
        peak_aqi=peak_aqi,
        peak_horizon=peak_horizon,
        peak_timestamp=peak_timestamp,
        peak_category=peak_category,
        first_advisory_horizon=first_advisory_h,
        first_advisory_timestamp=first_advisory_t,
        first_unhealthy_horizon=first_unhealthy_h,
        first_unhealthy_timestamp=first_unhealthy_t,
        first_very_unhealthy_horizon=first_very_unhealthy_h,
        first_very_unhealthy_timestamp=first_very_unhealthy_t,
        first_hazardous_horizon=first_hazardous_h,
        first_hazardous_timestamp=first_hazardous_t,
        advisory_horizon_count=advisory_count,
        unhealthy_horizon_count=unhealthy_count,
        very_unhealthy_horizon_count=very_unhealthy_count,
        hazardous_horizon_count=hazardous_count,
        severe_or_higher_horizon_count=severe_or_higher_count,
        title=title,
        message=message,
        data_is_stale=data_is_stale,
        fallback_active=fallback_active,
        feature_source=feature_source,
        upper_interval_crosses_hazardous=upper_crosses_haz,
    )
