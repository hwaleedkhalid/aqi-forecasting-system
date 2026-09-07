"""Pearls AQI Predictor - Post-Processing Layer.

Enriches raw model AQI predictions with EPA categories, official hex colors,
public health advisories, extreme event flags (>200 and >300), and
horizon-specific empirical prediction error intervals derived from
walk-forward out-of-fold residuals.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MODELS_DIR
from src.feature_pipeline.aqi_calculator import get_aqi_category
from src.inference.runtime_resolver import RuntimeAssetResolver
from src.logger import logger

# Official US EPA Health Advisories by Category
HEALTH_ADVISORIES: dict[str, str] = {
    "Good": "Air quality is satisfactory and air pollution poses little or no risk.",
    "Moderate": "Air quality is acceptable; unusually sensitive individuals should consider reducing prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Members of sensitive groups (children, elderly, respiratory/heart conditions) may experience health effects. General public is less likely to be affected.",
    "Unhealthy": "Some members of the general public may experience health effects; sensitive groups should avoid prolonged outdoor exertion.",
    "Very Unhealthy": "Health alert: The risk of health effects is increased for everyone. Avoid strenuous outdoor activities.",
    "Hazardous": "Health warning of emergency conditions: Everyone is more likely to be affected. Remain indoors and keep activity levels low.",
}


class AQIPostProcessor:
    """Enriches raw AQI predictions with domain metadata and empirical error intervals."""

    def __init__(
        self,
        intervals_path: Path | str | None = None,
        resolver: RuntimeAssetResolver | None = None,
    ) -> None:
        """Initialize post-processor and load empirical error intervals.

        Args:
            intervals_path: Path to empirical_error_intervals.json (default from RuntimeAssetResolver).
            resolver: Optional RuntimeAssetResolver instance.
        """
        self.resolver = resolver or RuntimeAssetResolver()
        self.intervals_path = Path(
            intervals_path or self.resolver.get_error_intervals_path()
        )
        self.q10: list[float] = []
        self.q90: list[float] = []
        self._load_error_intervals()

    def _load_error_intervals(self) -> None:
        """Load empirical residual quantiles (e_h = y_h - y_pred) from walk-forward validation."""
        if self.intervals_path.exists():
            try:
                with open(self.intervals_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.q10 = data.get("q10", [])
                self.q90 = data.get("q90", [])
                logger.info(
                    f"Loaded {len(self.q10)} empirical residual quantiles from {self.intervals_path}."
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load empirical error intervals from {self.intervals_path}: {e}")

        # Fallback approximation if file is missing (monotonically expanding empirical envelope)
        logger.info("Using default empirical residual quantiles fallback table.")
        # Residual quantiles widen with horizon
        self.q10 = [-25.0 - (i * 1.1) for i in range(72)]
        self.q90 = [20.0 + (i * 1.4) for i in range(72)]

    def get_advisory(self, category: str) -> str:
        """Get standard EPA health advisory text for a category."""
        return HEALTH_ADVISORIES.get(
            category,
            "Monitor local conditions and adjust outdoor activity as needed."
        )

    def compute_empirical_error_bounds(self, horizon: int, aqi: float) -> tuple[float, float]:
        """Compute empirical prediction error interval for horizon h.

        Formula:
          L_h = max(0.0, aqi + Q_0.10(e_h))
          U_h = aqi + Q_0.90(e_h)
        where e_h = y_h - y_pred from out-of-fold evaluations.

        Args:
            horizon: Horizon step (1-indexed, 1..72).
            aqi: Raw predicted AQI value.

        Returns:
            Tuple of (error_lower, error_upper).
        """
        idx = max(0, min(71, horizon - 1))
        q10_val = self.q10[idx] if idx < len(self.q10) else -50.0
        q90_val = self.q90[idx] if idx < len(self.q90) else 50.0

        error_lower = max(0.0, aqi + q10_val)
        error_upper = max(0.0, aqi + q90_val)
        return float(error_lower), float(error_upper)

    def process_horizon_point(
        self,
        horizon: int,
        target_time: datetime,
        raw_aqi: float,
    ) -> dict[str, Any]:
        """Format a single horizon forecast point into the API/UI contract.

        Args:
            horizon: Step number (1..72).
            target_time: Target timestamp for this horizon.
            raw_aqi: Non-negative predicted AQI value (not clipped to 500).

        Returns:
            Dictionary matching the forecast point contract.
        """
        aqi_val = float(max(0.0, raw_aqi))
        # Use existing EPA calculator for category and color
        # For AQI >= 301, this returns ('Hazardous', '#7E0023')
        category, color = get_aqi_category(aqi_val)
        advisory = self.get_advisory(category)

        err_lower, err_upper = self.compute_empirical_error_bounds(horizon, aqi_val)

        # Consistent Phase 11 threshold definitions
        high_severity = bool(aqi_val > 200.0)
        hazardous = bool(aqi_val > 300.0)

        return {
            "horizon": int(horizon),
            "forecast_time": target_time.isoformat(),
            "aqi": round(aqi_val, 1),
            "category": category,
            "color": color,
            "health_advisory": advisory,
            "error_lower": round(err_lower, 1),
            "error_upper": round(err_upper, 1),
            "high_severity": high_severity,
            "hazardous": hazardous,
        }

    def process_72h_forecast(
        self,
        raw_predictions: np.ndarray,
        forecast_origin: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Process 72-element predictions array into list of horizon forecast dicts.

        Args:
            raw_predictions: Array of shape (72,) containing non-negative predicted AQI values.
            forecast_origin: Timestamp of forecast origin (default: current UTC time).

        Returns:
            List of 72 horizon dictionaries.
        """
        origin = forecast_origin or datetime.now(timezone.utc)
        if origin.tzinfo is None:
            origin = origin.replace(tzinfo=timezone.utc)

        preds = np.asarray(raw_predictions).flatten()
        if len(preds) != 72:
            raise ValueError(f"Expected 72 prediction values, got {len(preds)}.")

        forecasts: list[dict[str, Any]] = []
        for h in range(1, 73):
            target_time = origin + timedelta(hours=h)
            point = self.process_horizon_point(h, target_time, float(preds[h - 1]))
            forecasts.append(point)

        return forecasts

    def build_summary(self, forecasts: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract high-level multi-horizon summary statistics from forecast points.

        Args:
            forecasts: List of 72 horizon dictionaries.

        Returns:
            Summary dictionary containing peak AQI, horizon of peak, and event alerts.
        """
        if not forecasts:
            return {}

        peak_point = max(forecasts, key=lambda p: p["aqi"])
        has_high_severity = any(p["high_severity"] for p in forecasts)
        has_hazardous = any(p["hazardous"] for p in forecasts)

        highest_alert = None
        if has_hazardous:
            highest_alert = "Hazardous (>300)"
        elif has_high_severity:
            highest_alert = "High Severity (>200)"

        return {
            "peak_aqi": peak_point["aqi"],
            "peak_horizon": peak_point["horizon"],
            "peak_category": peak_point["category"],
            "peak_forecast_time": peak_point["forecast_time"],
            "has_high_severity": has_high_severity,
            "has_hazardous": has_hazardous,
            "highest_alert_level": highest_alert,
        }
