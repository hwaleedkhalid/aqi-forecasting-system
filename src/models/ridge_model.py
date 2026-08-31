"""Pearls AQI Predictor - Ridge Regression Multi-Output Model.

Implements Ridge Regression wrapped in MultiOutputRegressor to forecast
72 independent hourly horizons from backward-looking pollutant features.
"""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from src.config import RANDOM_STATE, RIDGE_ALPHA
from src.exceptions import ModelTrainingError
from src.models.base_model import BaseAQIModel


class RidgeAQIModel(BaseAQIModel):
    """Multi-output Ridge regression forecasting model."""

    def __init__(
        self,
        alpha: float = RIDGE_ALPHA,
        random_state: int = RANDOM_STATE,
    ) -> None:
        """Initialize RidgeAQIModel.

        Args:
            alpha: L2 Regularization strength (default: 1.0).
            random_state: Random seed for deterministic reproducibility.
        """
        self.alpha = alpha
        self.random_state = random_state
        self.estimator = MultiOutputRegressor(
            Ridge(alpha=self.alpha, random_state=self.random_state)
        )
        self.is_fitted = False

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "Ridge Regression (MultiOutput)"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit 72 independent Ridge models on training features and targets.

        Args:
            X: Scaled feature matrix of shape (n_samples, n_features).
            y: Target matrix of shape (n_samples, forecast_horizons).
        """
        try:
            self.estimator.fit(X, y)
            self.is_fitted = True
        except Exception as err:
            raise ModelTrainingError(
                f"Failed fitting {self.name}: {err}", detail=str(err)
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 72 future hourly AQI values, clamped to valid range [0, 500].

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted AQI matrix of shape (n_samples, forecast_horizons).
        """
        if not self.is_fitted:
            raise ModelTrainingError("Model must be fitted before calling predict()")

        raw_preds = self.estimator.predict(X)
        # Clamp predictions to valid EPA AQI boundaries [0, 500]
        return np.clip(raw_preds, 0.0, 500.0)

    def get_coefficients(self, feature_names: list[str]) -> dict[str, list[float]]:
        """Extract linear coefficients for each horizon step.

        Args:
            feature_names: List of input feature column names.

        Returns:
            Dictionary mapping feature names to coefficient values across horizons.
        """
        if not self.is_fitted:
            raise ModelTrainingError("Model must be fitted before extracting coefficients")

        coefs_matrix = np.array([est.coef_ for est in self.estimator.estimators_])  # (72, n_features)
        return {
            feat: coefs_matrix[:, i].tolist()
            for i, feat in enumerate(feature_names)
        }

    def save(self, path: Path) -> Path:
        """Serialize fitted model to disk using joblib.

        Args:
            path: Destination file path.

        Returns:
            Path to saved artifact.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "RidgeAQIModel":
        """Load serialized model artifact from disk.

        Args:
            path: Path to model artifact.

        Returns:
            Loaded RidgeAQIModel instance.
        """
        model = joblib.load(path)
        if not isinstance(model, RidgeAQIModel):
            raise ModelTrainingError(
                f"Expected RidgeAQIModel instance from {path}, got {type(model)}"
            )
        return model
