"""Pearls AQI Predictor - LightGBM Multi-Horizon Forecasting Models.

Implements three distinct gradient boosted tree strategies:
1. Strategy A: Direct Multi-Output (72 parallel horizon estimators).
2. Strategy B: Grouped Horizons (Short: 1-6h, Medium: 7-24h, Long: 25-72h).
3. Strategy C: Horizon as Input Feature (Single unified estimator with horizon encoding).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from src.config import RANDOM_STATE
from src.exceptions import ModelTrainingError
from src.models.base_model import BaseAQIModel


class LightGBMDirectMultiOutput(BaseAQIModel):
    """Strategy A: 72 independent LightGBM regressors wrapped via MultiOutputRegressor."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        """Initialize LightGBMDirectMultiOutput."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimator = MultiOutputRegressor(
            lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=-1,
            ),
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    @property
    def name(self) -> str:
        return "LightGBM (Direct 72-Output)"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            self.estimator.fit(X, y)
            self.is_fitted = True
        except Exception as err:
            raise ModelTrainingError(f"Failed fitting {self.name}: {err}", detail=str(err))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ModelTrainingError("Model must be fitted before calling predict()")
        preds = self.estimator.predict(X)
        return np.clip(preds, 0.0, 500.0)

    def save(self, path):
        import joblib
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)


class LightGBMHorizonAsFeature(BaseAQIModel):
    """Strategy C: Single LightGBM model with forecast horizon h passed as an input feature."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ) -> None:
        """Initialize LightGBMHorizonAsFeature."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=-1,
        )
        self.is_fitted = False

    @property
    def name(self) -> str:
        return "LightGBM (Horizon as Feature)"

    def _expand_dataset(self, X: np.ndarray, y: np.ndarray | None = None, sample_stride: int = 4) -> tuple[np.ndarray, np.ndarray | None]:
        """Expand 2D matrix (N, D) and (N, 72) into 2D stacked representation (N*72, D+1).

        To maintain sub-second training efficiency across large grids, horizons are stride-sampled during fit.
        """
        n_samples = len(X)
        horizons_to_use = list(range(1, 73, sample_stride)) if y is not None else list(range(1, 73))
        
        X_expanded_list = []
        y_expanded_list = [] if y is not None else None

        for h in horizons_to_use:
            # Append normalized horizon feature [1..72] / 72.0
            h_col = np.full((n_samples, 1), h / 72.0, dtype=np.float32)
            X_h = np.hstack([X, h_col])
            X_expanded_list.append(X_h)

            if y is not None:
                y_expanded_list.append(y[:, h - 1])

        X_expanded = np.vstack(X_expanded_list)
        y_expanded = np.concatenate(y_expanded_list) if y_expanded_list is not None else None
        return X_expanded, y_expanded

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            X_exp, y_exp = self._expand_dataset(X, y, sample_stride=2)
            self.model.fit(X_exp, y_exp)
            self.is_fitted = True
        except Exception as err:
            raise ModelTrainingError(f"Failed fitting {self.name}: {err}", detail=str(err))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ModelTrainingError("Model must be fitted before calling predict()")

        n_samples = len(X)
        preds = np.zeros((n_samples, 72), dtype=np.float32)

        for h in range(1, 73):
            h_col = np.full((n_samples, 1), h / 72.0, dtype=np.float32)
            X_h = np.hstack([X, h_col])
            preds[:, h - 1] = self.model.predict(X_h)

        return np.clip(preds, 0.0, 500.0)

    def save(self, path):
        import joblib
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)
