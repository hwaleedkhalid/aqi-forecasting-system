"""Pearls AQI Predictor - Hybrid Specialist & Persistence-Aware Ensemble Models.

Implements:
1. HybridAQISpecialistModel:
   - High-performance LightGBM gradient boosted trees for high-frequency short horizons (h=1..6).
   - Stable, shrinkage-regularized Ridge Regression for medium/long horizons (h=7..72).
2. PersistenceAwareHybridModel:
   - LightGBM for short horizons (h=1..6).
   - Ridge Regression for intermediate horizons (h=7..37).
   - Smoothly blended Ridge + Persistence for long horizons (h=38..72) where physical decay converges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import Ridge

from src.models.base_model import BaseAQIModel
from src.models.lightgbm_models import LightGBMDirectMultiOutput


class HybridAQISpecialistModel(BaseAQIModel):
    """Combines LightGBM for short horizons (h1..6) and Ridge for medium/long horizons (h7..72)."""

    def __init__(
        self,
        short_estimators: int = 40,
        short_lr: float = 0.1,
        ridge_alpha: float = 1.0,
    ) -> None:
        """Initialize HybridAQISpecialistModel."""
        self._name = "Hybrid_LightGBM_Ridge_Specialist"
        self.m_short = LightGBMDirectMultiOutput(
            n_estimators=short_estimators, learning_rate=short_lr, num_leaves=20, n_jobs=-1
        )
        self.m_rest = Ridge(alpha=ridge_alpha, random_state=42)
        self.is_fitted = False

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> HybridAQISpecialistModel:
        """Fit short LightGBM model on h1..h6 and Ridge on h7..h72."""
        if y.shape[1] != 72:
            raise ValueError(f"Expected 72 target horizons, got shape {y.shape}")

        y_short = y[:, :6]
        y_rest = y[:, 6:]

        self.m_short.fit(X, y_short)
        self.m_rest.fit(X, y_rest)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, clip_max: float | None = 500.0) -> np.ndarray:
        """Predict composite 72 horizons."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        p_short = self.m_short.predict(X)
        p_rest = self.m_rest.predict(X)

        full_preds = np.hstack([p_short, p_rest])
        if clip_max is not None:
            return np.clip(full_preds, 0.0, clip_max)
        return np.maximum(0.0, full_preds)

    def save(self, path: Path) -> Path:
        """Serialize hybrid components."""
        import joblib
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"m_short": self.m_short, "m_rest": self.m_rest}, dest)
        return dest

    def load(self, path: Path) -> HybridAQISpecialistModel:
        """Load hybrid components."""
        import joblib
        data = joblib.load(path)
        self.m_short = data["m_short"]
        self.m_rest = data["m_rest"]
        self.is_fitted = True
        return self


class PersistenceAwareHybridModel(BaseAQIModel):
    """Hybrid specialist combining LightGBM (h1..6), Ridge (h7..37), and Blended Ridge+Persistence (h38..72)."""

    def __init__(
        self,
        short_estimators: int = 40,
        short_lr: float = 0.1,
        ridge_alpha: float = 1.0,
        min_blend_weight: float = 0.6,
        current_aqi_col_idx: int = 9,  # epa_aqi column index in feature matrix
    ) -> None:
        """Initialize PersistenceAwareHybridModel.

        Args:
            short_estimators: Tree count for short LightGBM model.
            short_lr: Learning rate for LightGBM.
            ridge_alpha: L2 regularizer for Ridge model.
            min_blend_weight: Minimum weight allocated to Ridge at h=72 (remaining allocated to persistence).
            current_aqi_col_idx: Feature column index containing current unscaled epa_aqi.
        """
        self._name = "Persistence_Aware_Hybrid_Specialist"
        self.m_short = LightGBMDirectMultiOutput(
            n_estimators=short_estimators, learning_rate=short_lr, num_leaves=20, n_jobs=-1
        )
        self.m_rest = Ridge(alpha=ridge_alpha, random_state=42)
        self.min_blend_weight = min_blend_weight
        self.current_aqi_col_idx = current_aqi_col_idx
        self.is_fitted = False

        # Precompute decay weights for h38..h72 (35 horizons)
        n_blend = 72 - 37  # 35
        # w_h smoothly transitions from 1.0 down to min_blend_weight
        self.blend_weights = np.linspace(1.0, self.min_blend_weight, n_blend)

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return self._name

    def fit(self, X: np.ndarray, y: np.ndarray) -> PersistenceAwareHybridModel:
        """Fit short LightGBM on h1..6 and Ridge on h7..72."""
        if y.shape[1] != 72:
            raise ValueError(f"Expected 72 target horizons, got shape {y.shape}")

        y_short = y[:, :6]
        y_rest = y[:, 6:]

        self.m_short.fit(X, y_short)
        self.m_rest.fit(X, y_rest)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        current_aqi: np.ndarray | None = None,
        clip_max: float | None = 500.0,
    ) -> np.ndarray:
        """Predict 72 horizons with persistence blending on horizons 38..72.

        Args:
            X: Feature matrix of shape (N, n_features).
            current_aqi: Optional 1D array of shape (N,) containing unscaled current AQI.
                         If None, will be extracted from X using current_aqi_col_idx.
            clip_max: Upper bound for prediction clipping. If None, predictions are
                      bounded only from below at 0.0, preserving extreme event fidelity.

        Returns:
            (N, 72) array of non-negative predictions.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")

        p_short = self.m_short.predict(X)        # Shape: (N, 6)
        p_rest = self.m_rest.predict(X)          # Shape: (N, 66) -> covers h7..h72

        p_med = p_rest[:, :31]                   # h7..h37 (31 horizons)
        p_long = p_rest[:, 31:].copy()           # h38..h72 (35 horizons)

        if current_aqi is None:
            # Fallback to column index if provided in raw features
            curr = X[:, self.current_aqi_col_idx]
        else:
            curr = current_aqi.flatten()

        curr_2d = curr[:, np.newaxis]            # Shape: (N, 1)

        # Apply persistence blending: w_h * p_long + (1 - w_h) * curr
        # blend_weights shape: (35,)
        blended_long = p_long * self.blend_weights + curr_2d * (1.0 - self.blend_weights)

        full_preds = np.hstack([p_short, p_med, blended_long])
        if clip_max is not None:
            return np.clip(full_preds, 0.0, clip_max)
        return np.maximum(0.0, full_preds)

    def save(self, path: Path) -> Path:
        """Serialize model."""
        import joblib
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "m_short": self.m_short,
                "m_rest": self.m_rest,
                "min_blend_weight": self.min_blend_weight,
                "current_aqi_col_idx": self.current_aqi_col_idx,
            },
            dest,
        )
        return dest

    def load(self, path: Path) -> PersistenceAwareHybridModel:
        """Load model."""
        import joblib
        data = joblib.load(path)
        self.m_short = data["m_short"]
        self.m_rest = data["m_rest"]
        self.min_blend_weight = data["min_blend_weight"]
        self.current_aqi_col_idx = data["current_aqi_col_idx"]
        self.is_fitted = True
        return self
