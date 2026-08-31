"""Pearls AQI Predictor - Random Forest Multi-Output Model.

Implements non-linear multi-target ensemble regression to forecast 72 future
hourly AQI horizons while modeling non-linear chemical interactions and thresholds.
"""

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import RANDOM_STATE, RF_MAX_DEPTH, RF_N_ESTIMATORS
from src.exceptions import ModelTrainingError
from src.models.base_model import BaseAQIModel


class RandomForestAQIModel(BaseAQIModel):
    """Multi-output Random Forest forecasting model for 72-hour AQI prediction."""

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: int | None = RF_MAX_DEPTH,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        """Initialize RandomForestAQIModel.

        Args:
            n_estimators: Number of trees in the forest (default: 100).
            max_depth: Maximum tree depth (default: None).
            random_state: Random seed for deterministic reproducibility.
            n_jobs: Number of parallel CPU worker threads (default: -1 for all cores).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "Random Forest Regressor"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest ensemble on multi-output targets.

        Args:
            X: Scaled feature matrix of shape (n_samples, n_features).
            y: Multi-output target matrix of shape (n_samples, forecast_horizons).
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
        # Clamp predictions to valid EPA AQI range [0, 500]
        return np.clip(raw_preds, 0.0, 500.0)

    def get_feature_importances(self, feature_names: list[str]) -> pd.DataFrame:
        """Extract global feature importances (mean impurity decrease).

        Args:
            feature_names: List of input feature column names.

        Returns:
            DataFrame sorted by importance score in descending order.
        """
        if not self.is_fitted:
            raise ModelTrainingError(
                "Model must be fitted before extracting feature importances"
            )

        importances = self.estimator.feature_importances_
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return df_imp

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
    def load(cls, path: Path) -> "RandomForestAQIModel":
        """Load serialized model artifact from disk.

        Args:
            path: Path to model artifact.

        Returns:
            Loaded RandomForestAQIModel instance.
        """
        model = joblib.load(path)
        if not isinstance(model, RandomForestAQIModel):
            raise ModelTrainingError(
                f"Expected RandomForestAQIModel instance from {path}, got {type(model)}"
            )
        return model
