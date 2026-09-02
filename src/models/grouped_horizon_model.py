"""Pearls AQI Predictor - Grouped Horizon Multi-Output Regressor.

Partitions the 72-hour forecasting horizon into three temporal regimes:
1. Short Horizon (h=1 to h=6): captures high-frequency dispersion and short-term atmospheric persistence.
2. Medium Horizon (h=7 to h=24): captures diurnal cycles, solar radiative forcing, and nightly inversion.
3. Long Horizon (h=25 to h=72): captures multi-day synoptic weather systems and baseline seasonal trends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import numpy as np
from sklearn.linear_model import Ridge

from src.models.base_model import BaseAQIModel


class GroupedHorizonModel(BaseAQIModel):
    """Regresses over distinct temporal groups with independently trained multi-output estimators."""

    def __init__(
        self,
        h1_6_estimator: Any | None = None,
        h7_24_estimator: Any | None = None,
        h25_72_estimator: Any | None = None,
    ) -> None:
        """Initialize GroupedHorizonModel.

        Args:
            h1_6_estimator: Estimator for horizons 1..6. Defaults to Ridge(alpha=1.0).
            h7_24_estimator: Estimator for horizons 7..24. Defaults to Ridge(alpha=1.0).
            h25_72_estimator: Estimator for horizons 25..72. Defaults to Ridge(alpha=1.0).
        """
        self._name = "Grouped_Horizon_Model"
        self.m_short = h1_6_estimator or Ridge(alpha=1.0, random_state=42)
        self.m_medium = h7_24_estimator or Ridge(alpha=1.0, random_state=42)
        self.m_long = h25_72_estimator or Ridge(alpha=1.0, random_state=42)
        self.is_fitted = False

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> GroupedHorizonModel:
        """Fit independent estimators on the partitioned horizon target slices.

        Args:
            X: Feature matrix of shape (N, n_features).
            y: Target matrix of shape (N, 72).

        Returns:
            Fitted instance.
        """
        if y.shape[1] != 72:
            raise ValueError(f"Expected 72 target horizons, got shape {y.shape}")

        y_short = y[:, :6]      # h1..h6 (6 columns)
        y_medium = y[:, 6:24]   # h7..h24 (18 columns)
        y_long = y[:, 24:]      # h25..h72 (48 columns)

        self.m_short.fit(X, y_short)
        self.m_medium.fit(X, y_medium)
        self.m_long.fit(X, y_long)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 72 horizons by concatenating predictions from each specialized group.

        Args:
            X: Feature matrix of shape (N, n_features).

        Returns:
            (N, 72) array of predictions clipped to [0, 500].
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")

        p_short = self.m_short.predict(X)
        if p_short.ndim == 1:
            p_short = p_short[:, np.newaxis]

        p_medium = self.m_medium.predict(X)
        if p_medium.ndim == 1:
            p_medium = p_medium[:, np.newaxis]

        p_long = self.m_long.predict(X)
        if p_long.ndim == 1:
            p_long = p_long[:, np.newaxis]

        full_preds = np.hstack([p_short, p_medium, p_long])
        return np.clip(full_preds, 0.0, 500.0)

    def save(self, path: Path) -> Path:
        """Serialize grouped models."""
        import joblib
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"m_short": self.m_short, "m_medium": self.m_medium, "m_long": self.m_long},
            dest,
        )
        return dest

    def load(self, path: Path) -> GroupedHorizonModel:
        """Load grouped models."""
        import joblib
        data = joblib.load(path)
        self.m_short = data["m_short"]
        self.m_medium = data["m_medium"]
        self.m_long = data["m_long"]
        self.is_fitted = True
        return self
