"""Pearls AQI Predictor - Chronological Training-Only Cross-Validator.

Executes temporal expanding-window cross-validation exclusively on the training partition,
preventing data leakage and overfitting while evaluating models across multiple historical regimes.
"""

from __future__ import annotations

import time
from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.logger import logger


class ChronologicalCVEvaluator:
    """Manages expanding-window chronological cross-validation on training data."""

    def __init__(
        self,
        n_folds: int = 3,
        min_train_fraction: float = 0.55,
        val_fraction_per_fold: float = 0.15,
    ) -> None:
        """Initialize ChronologicalCVEvaluator.

        Args:
            n_folds: Number of expanding-window temporal folds.
            min_train_fraction: Proportion of training data used for initial fold training.
            val_fraction_per_fold: Proportion of training data evaluated in each validation fold.
        """
        self.n_folds = n_folds
        self.min_train_fraction = min_train_fraction
        self.val_fraction_per_fold = val_fraction_per_fold

    def get_folds(
        self,
        n_samples: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate expanding-window train and validation index splits.

        Args:
            n_samples: Total number of training samples.

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        folds = []
        step_size = (1.0 - self.min_train_fraction) / self.n_folds

        for fold in range(self.n_folds):
            train_end_ratio = self.min_train_fraction + fold * step_size
            val_end_ratio = min(1.0, train_end_ratio + self.val_fraction_per_fold)

            train_end_idx = int(n_samples * train_end_ratio)
            val_end_idx = int(n_samples * val_end_ratio)

            train_idx = np.arange(0, train_end_idx)
            val_idx = np.arange(train_end_idx, val_end_idx)

            folds.append((train_idx, val_idx))

        return folds

    def evaluate_model_cv(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
        predict_fn: Callable[[Any, np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Train and evaluate a model across all chronological folds.

        Args:
            model_factory: Factory function returning a fresh model instance for each fold.
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target matrix of shape (n_samples, forecast_horizons).
            predict_fn: Optional custom prediction function (default uses model.predict).

        Returns:
            Dictionary containing averaged metrics (mean, std) and per-horizon benchmarks.
        """
        n_samples = len(X)
        folds = self.get_folds(n_samples)

        fold_rmses: list[float] = []
        fold_maes: list[float] = []
        fold_r2s: list[float] = []
        fold_h1: list[float] = []
        fold_h24: list[float] = []
        fold_h72: list[float] = []

        start_time = time.perf_counter()

        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # Initialize fresh model and fit
            model = model_factory()
            model.fit(X_train_fold, y_train_fold)

            # Generate predictions
            if predict_fn is not None:
                preds = predict_fn(model, X_val_fold)
            else:
                preds = model.predict(X_val_fold)

            preds = np.clip(preds, 0.0, 500.0)

            # Compute fold metrics
            rmse = float(np.sqrt(mean_squared_error(y_val_fold, preds)))
            mae = float(mean_absolute_error(y_val_fold, preds))
            r2 = float(r2_score(y_val_fold, preds))

            h1_rmse = float(np.sqrt(mean_squared_error(y_val_fold[:, 0], preds[:, 0])))
            h24_rmse = float(np.sqrt(mean_squared_error(y_val_fold[:, 23], preds[:, 23])))
            h72_rmse = float(np.sqrt(mean_squared_error(y_val_fold[:, 71], preds[:, 71])))

            fold_rmses.append(rmse)
            fold_maes.append(mae)
            fold_r2s.append(r2)
            fold_h1.append(h1_rmse)
            fold_h24.append(h24_rmse)
            fold_h72.append(h72_rmse)

        elapsed_time = time.perf_counter() - start_time

        results = {
            "val_rmse_mean": float(np.mean(fold_rmses)),
            "val_rmse_std": float(np.std(fold_rmses)),
            "val_mae_mean": float(np.mean(fold_maes)),
            "val_r2_mean": float(np.mean(fold_r2s)),
            "val_h1_rmse": float(np.mean(fold_h1)),
            "val_h24_rmse": float(np.mean(fold_h24)),
            "val_h72_rmse": float(np.mean(fold_h72)),
            "fold_rmses": fold_rmses,
            "training_time_sec": float(elapsed_time),
        }
        return results
