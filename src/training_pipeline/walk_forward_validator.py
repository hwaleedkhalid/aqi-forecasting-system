"""Pearls AQI Predictor - Phase 11: Walk-Forward Stability & Diagnostic Validator.

Modular architecture:
  WalkForwardFoldSpec       — immutable dataclass defining fold boundaries
  WalkForwardFoldEngine     — constructs folds and verifies embargo invariant
  FoldTrainer               — fits scaler + model from scratch on fold training data
  DiagnosticEvaluator       — computes all 5 diagnostic dimensions
  WalkForwardReportGenerator — serializes JSON + CSV reports

Key invariant enforced:
  max(train_feature_timestamp) + FORECAST_HORIZONS hours + 1h <= min(val_feature_timestamp)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

from src.config import FORECAST_HORIZONS, MODELS_DIR
from src.logger import logger
from src.models.hybrid_specialist_model import (
    HybridAQISpecialistModel,
    PersistenceAwareHybridModel,
)
from src.models.ridge_model import RidgeAQIModel
from src.training_pipeline.season_classifier import LahoreSeasonClassifier


# =============================================================================
# Fold Specification
# =============================================================================

@dataclass(frozen=True)
class WalkForwardFoldSpec:
    """Immutable specification for one walk-forward fold.

    Attributes:
        fold_id: Human-readable fold identifier (e.g. 'fold_1_winter2021').
        train_end: Inclusive end timestamp for training features (UTC).
        val_start: Inclusive start timestamp for validation features (UTC).
        val_end: Inclusive end timestamp for validation features (UTC).
        embargo_hours: Number of hours required between training and validation.
        primary_seasons: Informational note on what seasons appear in this val window.
    """
    fold_id: str
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    embargo_hours: int = FORECAST_HORIZONS
    primary_seasons: str = ""


# =============================================================================
# Fold Engine
# =============================================================================

class WalkForwardFoldEngine:
    """Constructs and validates walk-forward fold specifications.

    Verifies the embargo invariant at fold-construction time:
        max(train_feature_timestamp) + embargo_hours hours + 1h <= min(val_feature_timestamp)

    This guarantees that no training target timestamp can overlap
    with any validation input timestamp.
    """

    FOLD_SPECS: list[dict[str, Any]] = [
        {
            "fold_id": "fold_1_winter2021",
            "train_end": "2021-10-28 00:00:00+00:00",
            "val_start": "2021-11-01 01:00:00+00:00",
            "val_end":   "2022-01-31 23:00:00+00:00",
            "primary_seasons": "winter_smog",
        },
        {
            "fold_id": "fold_2_transition_summer2022",
            "train_end": "2022-03-28 00:00:00+00:00",
            "val_start": "2022-04-01 01:00:00+00:00",
            "val_end":   "2022-06-30 23:00:00+00:00",
            "primary_seasons": "transition, summer",
        },
        {
            "fold_id": "fold_3_monsoon2022",
            "train_end": "2022-06-27 00:00:00+00:00",
            "val_start": "2022-07-01 01:00:00+00:00",
            "val_end":   "2022-09-30 23:00:00+00:00",
            "primary_seasons": "monsoon",
        },
        {
            "fold_id": "fold_4_winter2023",
            "train_end": "2023-10-28 00:00:00+00:00",
            "val_start": "2023-11-01 01:00:00+00:00",
            "val_end":   "2024-01-31 23:00:00+00:00",
            "primary_seasons": "winter_smog (2-year gap from Fold 1)",
        },
    ]

    def build_folds(self) -> list[WalkForwardFoldSpec]:
        """Build and validate all fold specifications."""
        folds = []
        for spec in self.FOLD_SPECS:
            fold = WalkForwardFoldSpec(
                fold_id=spec["fold_id"],
                train_end=pd.Timestamp(spec["train_end"]),
                val_start=pd.Timestamp(spec["val_start"]),
                val_end=pd.Timestamp(spec["val_end"]),
                embargo_hours=FORECAST_HORIZONS,
                primary_seasons=spec["primary_seasons"],
            )
            self._verify_embargo_invariant(fold)
            folds.append(fold)
        return folds

    def _verify_embargo_invariant(self, fold: WalkForwardFoldSpec) -> None:
        """Verify that the embargo invariant holds for a fold.

        Invariant: max(train_feature_timestamp) + embargo_hours + 1h <= val_start

        Since training features generate targets up to train_end + embargo_hours,
        we require val_start > train_end + embargo_hours.

        Raises:
            ValueError: If the embargo invariant is violated.
        """
        required_gap_end = fold.train_end + pd.Timedelta(hours=fold.embargo_hours)
        if fold.val_start <= required_gap_end:
            raise ValueError(
                f"Embargo invariant violated for {fold.fold_id}: "
                f"val_start ({fold.val_start}) must be > "
                f"train_end + embargo_hours ({required_gap_end}). "
                f"Actual gap: {fold.val_start - fold.train_end}"
            )
        logger.info(
            f"[{fold.fold_id}] Embargo verified: train_end={fold.train_end}, "
            f"val_start={fold.val_start}, gap={fold.val_start - fold.train_end}"
        )


# =============================================================================
# Fold Trainer (refits all preprocessing from scratch per fold)
# =============================================================================

class FoldTrainer:
    """Fits scaler and model exclusively on one fold's training data."""

    def __init__(self, model_factory, feature_cols: list[str]) -> None:
        self.model_factory = model_factory
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.model = None

    def fit(
        self,
        df_features: pd.DataFrame,
        df_targets: pd.DataFrame,
        fold: WalkForwardFoldSpec,
        current_aqi_col: str = "epa_aqi",
    ) -> None:
        """Fit scaler and model on training portion of this fold.

        Args:
            df_features: Feature dataframe with 'datetime_utc' column.
            df_targets: Target dataframe with 'datetime_utc' and target columns.
            fold: Fold spec defining training boundary.
            current_aqi_col: Column name for current AQI (used in naive baseline).
        """
        train_mask = df_features["datetime_utc"] <= fold.train_end
        X_train_raw = df_features.loc[train_mask, self.feature_cols].values.astype(np.float32)
        y_train = df_targets.loc[train_mask].values.astype(np.float32)

        X_train = self.scaler.fit_transform(X_train_raw)

        self.model = self.model_factory()
        self.model.fit(X_train, y_train)
        logger.info(f"[{fold.fold_id}] Trained on {X_train.shape[0]} samples.")

    def predict(
        self,
        df_features: pd.DataFrame,
        fold: WalkForwardFoldSpec,
        current_aqi: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate predictions on the validation window of this fold.

        Args:
            df_features: Feature dataframe with 'datetime_utc' column.
            fold: Fold spec defining validation boundary.
            current_aqi: Optional 1D array of unscaled AQI for persistence-aware models.

        Returns:
            (N_val, 72) array of predictions.
        """
        val_mask = (
            (df_features["datetime_utc"] >= fold.val_start)
            & (df_features["datetime_utc"] <= fold.val_end)
        )
        X_val_raw = df_features.loc[val_mask, self.feature_cols].values.astype(np.float32)
        X_val = self.scaler.transform(X_val_raw)

        if hasattr(self.model, "predict") and isinstance(self.model, PersistenceAwareHybridModel):
            return self.model.predict(X_val, current_aqi=current_aqi)
        return self.model.predict(X_val)


# =============================================================================
# Diagnostic Evaluator
# =============================================================================

LOW_SAMPLE_THRESHOLD = 30
season_clf = LahoreSeasonClassifier()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def _extreme_event_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float, label: str
) -> dict[str, Any]:
    """Compute extreme-event metrics with sample count and low-n flag."""
    mask = y_true > threshold
    n = int(mask.sum())
    low_n = n < LOW_SAMPLE_THRESHOLD
    if n == 0:
        return {
            "threshold": threshold,
            "label": label,
            "n": 0,
            "rmse": None,
            "mae": None,
            "low_sample_warning": True,
        }
    return {
        "threshold": threshold,
        "label": label,
        "n": n,
        "rmse": round(_rmse(y_true[mask], y_pred[mask]), 3),
        "mae": round(_mae(y_true[mask], y_pred[mask]), 3),
        "low_sample_warning": low_n,
    }


class DiagnosticEvaluator:
    """Computes all 5 diagnostic dimensions for a fold's predictions."""

    HORIZON_GROUPS = {
        "short_h1_6": (0, 6),
        "medium_h7_24": (6, 24),
        "medium_long_h25_48": (24, 48),
        "long_h49_72": (48, 72),
    }

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        val_timestamps: pd.Series,
        df_train_features: pd.DataFrame,
        df_val_features: pd.DataFrame,
        distribution_cols: list[str],
    ) -> dict[str, Any]:
        """Run all 5 diagnostic dimensions.

        Args:
            y_true: (N, 72) ground truth target matrix.
            y_pred: (N, 72) model predictions.
            val_timestamps: Series of UTC timestamps for each validation sample.
            df_train_features: Full training feature dataframe (for distribution shift).
            df_val_features: Validation feature dataframe (for distribution shift).
            distribution_cols: Feature column names to include in distribution shift analysis.

        Returns:
            Dictionary with keys: overall, seasonal, horizon_groups, extreme_events, distribution_shift.
        """
        return {
            "overall": self._overall_metrics(y_true, y_pred),
            "seasonal": self._seasonal_metrics(y_true, y_pred, val_timestamps),
            "horizon_groups": self._horizon_metrics(y_true, y_pred),
            "extreme_events": self._extreme_event_metrics(y_true, y_pred),
            "distribution_shift": self._distribution_shift(
                df_train_features, df_val_features, distribution_cols
            ),
        }

    def _overall_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": round(_rmse(y_true, y_pred), 3),
            "mae": round(_mae(y_true, y_pred), 3),
            "r2": round(_r2(y_true, y_pred), 4),
            "n_samples": int(y_true.shape[0]),
        }

    def _seasonal_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, val_timestamps: pd.Series
    ) -> dict[str, Any]:
        months = val_timestamps.dt.month.values
        results: dict[str, Any] = {}
        for month_val, season in {m: season_clf.classify(m) for m in set(months)}.items():
            mask = months == month_val
        # Group by season directly
        season_labels = np.array([season_clf.classify(int(m)) for m in months])
        for season in sorted(set(season_labels)):
            s_mask = season_labels == season
            n = int(s_mask.sum())
            if n == 0:
                continue
            results[season] = {
                "n": n,
                "rmse": round(_rmse(y_true[s_mask], y_pred[s_mask]), 3),
                "mae": round(_mae(y_true[s_mask], y_pred[s_mask]), 3),
                "r2": round(_r2(y_true[s_mask], y_pred[s_mask]), 4),
                "low_sample_warning": n < LOW_SAMPLE_THRESHOLD,
            }
        return results

    def _horizon_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for group_name, (h_start, h_end) in self.HORIZON_GROUPS.items():
            y_t = y_true[:, h_start:h_end]
            y_p = y_pred[:, h_start:h_end]
            results[group_name] = {
                "horizons": f"h{h_start+1}-h{h_end}",
                "rmse": round(_rmse(y_t, y_p), 3),
                "mae": round(_mae(y_t, y_p), 3),
                "r2": round(_r2(y_t, y_p), 4),
            }
        return results

    def _extreme_event_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, Any]:
        # Top 1% absolute errors
        abs_err = np.abs(y_true - y_pred).flatten()
        threshold_top1 = float(np.percentile(abs_err, 99))
        top1_mask = abs_err >= threshold_top1
        n_top1 = int(top1_mask.sum())

        return {
            "high_severity_gt200": _extreme_event_metrics(y_true, y_pred, 200.0, "AQI>200"),
            "hazardous_gt300": _extreme_event_metrics(y_true, y_pred, 300.0, "AQI>300"),
            "top_1pct_errors": {
                "threshold_abs_error": round(threshold_top1, 2),
                "n": n_top1,
                "mean_abs_error": round(float(abs_err[top1_mask].mean()), 3),
                "low_sample_warning": n_top1 < LOW_SAMPLE_THRESHOLD,
            },
        }

    def _distribution_shift(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        cols: list[str],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for col in cols:
            if col not in df_train.columns or col not in df_val.columns:
                continue
            train_vals = df_train[col].dropna().values
            val_vals = df_val[col].dropna().values
            if len(train_vals) == 0 or len(val_vals) == 0:
                continue

            # Compute Wasserstein distance as scalar shift statistic
            w_dist = float(wasserstein_distance(train_vals, val_vals))

            results[col] = {
                "train_median": round(float(np.median(train_vals)), 3),
                "train_p90": round(float(np.percentile(train_vals, 90)), 3),
                "train_p99": round(float(np.percentile(train_vals, 99)), 3),
                "val_median": round(float(np.median(val_vals)), 3),
                "val_p90": round(float(np.percentile(val_vals, 90)), 3),
                "val_p99": round(float(np.percentile(val_vals, 99)), 3),
                "wasserstein_distance": round(w_dist, 4),
            }

        # Event frequency
        for col, thresh, label in [
            ("epa_aqi", 200.0, "high_severity_freq_gt200"),
            ("epa_aqi", 300.0, "hazardous_freq_gt300"),
        ]:
            if col in df_train.columns and col in df_val.columns:
                results[label] = {
                    "train_fraction": round(float((df_train[col] > thresh).mean()), 4),
                    "val_fraction": round(float((df_val[col] > thresh).mean()), 4),
                }

        return results


# =============================================================================
# Report Generator
# =============================================================================

class WalkForwardReportGenerator:
    """Compiles fold results into JSON and CSV outputs."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json_report(self, results: dict[str, Any]) -> Path:
        path = self.output_dir / "walk_forward_report.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved walk-forward JSON report to {path}")
        return path

    def save_summary_csv(self, results: dict[str, Any]) -> Path:
        """Build a flat 16-row (4 models × 4 folds) summary CSV."""
        rows = []
        for fold_id, fold_data in results["folds"].items():
            for model_key, model_data in fold_data["models"].items():
                overall = model_data["diagnostics"]["overall"]
                hor = model_data["diagnostics"]["horizon_groups"]
                ext = model_data["diagnostics"]["extreme_events"]
                rows.append({
                    "fold_id": fold_id,
                    "model_key": model_key,
                    "primary_seasons": fold_data.get("primary_seasons", ""),
                    "n_samples": overall.get("n_samples", 0),
                    "overall_rmse": overall.get("rmse"),
                    "overall_mae": overall.get("mae"),
                    "overall_r2": overall.get("r2"),
                    "short_h1_6_rmse": hor.get("short_h1_6", {}).get("rmse"),
                    "medium_h7_24_rmse": hor.get("medium_h7_24", {}).get("rmse"),
                    "medium_long_h25_48_rmse": hor.get("medium_long_h25_48", {}).get("rmse"),
                    "long_h49_72_rmse": hor.get("long_h49_72", {}).get("rmse"),
                    "high_severity_gt200_n": ext.get("high_severity_gt200", {}).get("n"),
                    "high_severity_gt200_rmse": ext.get("high_severity_gt200", {}).get("rmse"),
                    "hazardous_gt300_n": ext.get("hazardous_gt300", {}).get("n"),
                    "hazardous_gt300_rmse": ext.get("hazardous_gt300", {}).get("rmse"),
                    "hazardous_gt300_low_n_flag": ext.get("hazardous_gt300", {}).get("low_sample_warning"),
                })
        df = pd.DataFrame(rows)
        path = self.output_dir / "walk_forward_summary.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved walk-forward summary CSV to {path}")
        return path

    def build_stability_report(self, results: dict[str, Any]) -> dict[str, Any]:
        """Compute EXP-019 vs Naive stability statistics across folds."""
        exp019_rmses, naive_rmses = [], []
        for fold_id, fold_data in results["folds"].items():
            models = fold_data["models"]
            if "exp019" in models and "naive" in models:
                exp019_rmses.append(models["exp019"]["diagnostics"]["overall"]["rmse"])
                naive_rmses.append(models["naive"]["diagnostics"]["overall"]["rmse"])

        deltas = [n - e for n, e in zip(naive_rmses, exp019_rmses)]  # positive = EXP-019 wins
        if not deltas:
            return {}

        return {
            "fold_count": len(deltas),
            "exp019_wins": int(sum(d > 0 for d in deltas)),
            "deltas_per_fold": {f"fold_{i+1}": round(d, 3) for i, d in enumerate(deltas)},
            "mean_delta": round(float(np.mean(deltas)), 3),
            "median_delta": round(float(np.median(deltas)), 3),
            "std_delta": round(float(np.std(deltas)), 3),
            "worst_fold_delta": round(float(min(deltas)), 3),
            "best_fold_delta": round(float(max(deltas)), 3),
            "exp019_rmse_mean": round(float(np.mean(exp019_rmses)), 3),
            "naive_rmse_mean": round(float(np.mean(naive_rmses)), 3),
            "relative_gain_percent": round(
                (np.mean(naive_rmses) - np.mean(exp019_rmses)) / np.mean(naive_rmses) * 100, 2
            ),
        }
