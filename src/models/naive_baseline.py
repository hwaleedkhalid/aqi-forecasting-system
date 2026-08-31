"""Pearls AQI Predictor - Naive Persistence Baseline Model.

Implements the mandatory persistence benchmark:
    y_hat_{t+h} = y_t   for all h in [1, 72]

Every machine learning model must achieve lower RMSE/MAE and higher R^2
than this zero-parameter persistence baseline.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np

from src.config import FORECAST_HORIZONS
from src.exceptions import ValidationError
from src.models.base_model import BaseAQIModel


class NaivePersistenceBaseline(BaseAQIModel):
    """Persistence baseline forecasting the current AQI value forward across all horizons."""

    def __init__(self, forecast_horizons: int = FORECAST_HORIZONS) -> None:
        """Initialize NaivePersistenceBaseline.

        Args:
            forecast_horizons: Number of forecast steps to repeat current AQI (default: 72).
        """
        self.forecast_horizons = forecast_horizons
        self._is_fitted = True

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "Naive Persistence Baseline"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """No-op fit for zero-parameter persistence baseline."""
        self._is_fitted = True

    def predict_from_current(self, current_aqi: np.ndarray) -> np.ndarray:
        """Generate 72-hour forecast by broadcasting current unscaled ground-truth AQI.

        Args:
            current_aqi: 1D array of shape (N,) containing unscaled AQI_t.

        Returns:
            Forecast matrix of shape (N, forecast_horizons).
        """
        arr = np.asarray(current_aqi).ravel()
        # Broadcast (N, 1) -> (N, 72)
        return np.repeat(arr[:, np.newaxis], self.forecast_horizons, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions.

        If X is 1D or single-column (N, 1), treats it as current AQI.
        Otherwise, expects X to contain current AQI or raises descriptive ValidationError.

        Args:
            X: Input array of current AQI values.

        Returns:
            Forecast matrix of shape (N, forecast_horizons).
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            return self.predict_from_current(X_arr)
        elif X_arr.ndim == 2 and X_arr.shape[1] == 1:
            return self.predict_from_current(X_arr.ravel())
        else:
            raise ValidationError(
                "NaivePersistenceBaseline.predict expects 1D array of current AQI values",
                detail=f"Received input with shape {X_arr.shape}. Use predict_from_current() instead.",
            )

    def save(self, path: Path) -> Path:
        """Save baseline configuration metadata."""
        meta = {
            "name": self.name,
            "forecast_horizons": self.forecast_horizons,
            "type": "persistence",
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return path

    def load(self, path: Path) -> "NaivePersistenceBaseline":
        """Load baseline configuration metadata."""
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.forecast_horizons = meta.get("forecast_horizons", self.forecast_horizons)
        return self
