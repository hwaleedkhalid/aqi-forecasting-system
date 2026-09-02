"""Pearls AQI Predictor - Multi-Pollutant to EPA AQI Forecasting Model.

Implements the physical forecasting formulation:
1. Multi-output models forecast continuous concentrations for all 6 US EPA criteria pollutants:
   (PM2.5, PM10, O3, NO2, SO2, CO) across 72 prediction horizons.
2. Predictions are converted to individual EPA AQI sub-indices using vectorized piecewise linear interpolation.
3. Overall AQI is derived as the elementwise maximum across all 6 pollutant sub-indices:
   AQI(t+h) = max(I_PM2.5, I_PM10, I_O3, I_NO2, I_SO2, I_CO)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import numpy as np
from sklearn.linear_model import Ridge

from src.feature_pipeline.aqi_calculator import EPA_BREAKPOINTS
from src.logger import logger
from src.models.base_model import BaseAQIModel

CRITERIA_POLLUTANTS = ["pm2_5", "pm10", "o3", "no2", "so2", "co"]


def calculate_vectorized_sub_index(pollutant: str, conc_array: np.ndarray) -> np.ndarray:
    """Calculate US EPA AQI sub-index for a 2D numpy array of pollutant concentrations.

    Formula:
        I = ((I_high - I_low) / (C_high - C_low)) * (C_p - C_low) + I_low

    Args:
        pollutant: Pollutant identifier (e.g. 'pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co').
        conc_array: 2D numpy array of concentrations in μg/m³ of shape (N, H).

    Returns:
        2D numpy array of integer/float AQI sub-indices in range [0, 500].
    """
    breakpoints = EPA_BREAKPOINTS.get(pollutant.lower())
    if not breakpoints:
        raise ValueError(f"Unknown pollutant for EPA calculation: {pollutant}")

    # Clip negative values to zero
    c_p = np.maximum(0.0, conc_array.astype(np.float64))
    sub_index_matrix = np.zeros_like(c_p)

    # Apply piecewise linear interpolation for each breakpoint tier
    for idx, (c_low, c_high, i_low, i_high) in enumerate(breakpoints):
        mask = (c_p >= c_low) & (c_p <= c_high)
        slope = (i_high - i_low) / (c_high - c_low)
        sub_index_matrix[mask] = slope * (c_p[mask] - c_low) + i_low

    # Handle concentrations above the highest defined breakpoint (> C_high of tier 7)
    highest_c_high = breakpoints[-1][1]
    sub_index_matrix[c_p > highest_c_high] = 500.0

    return np.clip(sub_index_matrix, 0.0, 500.0)


def calculate_vectorized_epa_aqi(pollutant_predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Calculate overall EPA AQI as the elementwise maximum across all criteria pollutant sub-indices.

    Args:
        pollutant_predictions: Dictionary mapping pollutant name -> (N, H) predicted concentration array.

    Returns:
        (N, H) array representing overall EPA AQI forecasts.
    """
    sub_indices_list: list[np.ndarray] = []

    for pol in CRITERIA_POLLUTANTS:
        if pol in pollutant_predictions:
            sub_idx = calculate_vectorized_sub_index(pol, pollutant_predictions[pol])
            sub_indices_list.append(sub_idx)

    if not sub_indices_list:
        raise ValueError("No valid pollutant predictions provided to compute EPA AQI.")

    # Stack along new axis and take max along axis 0
    stacked = np.stack(sub_indices_list, axis=0)  # Shape: (6, N, H)
    overall_aqi = np.max(stacked, axis=0)  # Shape: (N, H)
    return np.clip(overall_aqi, 0.0, 500.0)


class MultiPollutantToAQIModel(BaseAQIModel):
    """Forecasting architecture predicting 6 pollutant concentrations before EPA AQI conversion."""

    def __init__(
        self,
        estimator_factory: Callable[[], Any] | None = None,
        pollutant_models: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MultiPollutantToAQIModel.

        Args:
            estimator_factory: Callable returning a scikit-learn compatible multi-output regressor.
                               Defaults to Ridge(alpha=1.0).
            pollutant_models: Optional pre-configured dictionary mapping pollutant name -> estimator.
        """
        self._name = "MultiPollutant_to_EPA_AQI"
        self.estimator_factory = estimator_factory or (lambda: Ridge(alpha=1.0, random_state=42))
        self.models: dict[str, Any] = pollutant_models or {
            pol: self.estimator_factory() for pol in CRITERIA_POLLUTANTS
        }
        self.is_fitted = False

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self._name

    def fit(self, X: np.ndarray, y: Any) -> MultiPollutantToAQIModel:
        """Fit individual multi-output estimators for each criteria pollutant.

        Args:
            X: Feature matrix of shape (N, n_features).
            y: Dictionary mapping each criteria pollutant ('pm2_5', 'pm10', 'o3', 'no2', 'so2', 'co')
               to a multi-horizon target matrix of shape (N, 72).

        Returns:
            Fitted instance.
        """
        if not isinstance(y, dict):
            raise TypeError("MultiPollutantToAQIModel expects a dictionary mapping pollutant -> (N, 72) target array.")

        for pol in CRITERIA_POLLUTANTS:
            if pol not in y:
                raise ValueError(f"Target dictionary missing required pollutant: {pol}")
            self.models[pol].fit(X, y[pol])

        self.is_fitted = True
        return self

    def predict_pollutants(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Predict multi-horizon concentrations for all 6 criteria pollutants.

        Args:
            X: Feature matrix of shape (N, n_features).

        Returns:
            Dictionary mapping pollutant name -> (N, 72) array of predicted concentrations.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_pollutants.")

        predictions: dict[str, np.ndarray] = {}
        for pol in CRITERIA_POLLUTANTS:
            raw_preds = self.models[pol].predict(X)
            # Clip physical concentrations to non-negative
            predictions[pol] = np.maximum(0.0, raw_preds)

        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict overall EPA AQI by converting multi-pollutant forecasts.

        Args:
            X: Feature matrix of shape (N, n_features).

        Returns:
            (N, 72) array of predicted EPA AQI values clamped to [0, 500].
        """
        pollutant_preds = self.predict_pollutants(X)
        return calculate_vectorized_epa_aqi(pollutant_preds)

    def save(self, path: Path) -> Path:
        """Serialize pollutant models to disk."""
        import joblib
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models, dest)
        return dest

    def load(self, path: Path) -> MultiPollutantToAQIModel:
        """Load pollutant models from disk."""
        import joblib
        self.models = joblib.load(path)
        self.is_fitted = True
        return self
