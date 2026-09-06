"""Pearls AQI Predictor - Phase 11: Walk-Forward Stability Experiment Runner.

Loads enriched features and targets, runs 4 × 4 = 16 model-fold evaluations,
and saves comprehensive stability/diagnostic reports to data/models/walk_forward/.

Usage:
    python -m src.training_pipeline.run_walk_forward
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.logger import logger
from src.models.hybrid_specialist_model import (
    HybridAQISpecialistModel,
    PersistenceAwareHybridModel,
)
from src.models.naive_baseline import NaivePersistenceBaseline
from src.models.ridge_model import RidgeAQIModel
from src.training_pipeline.walk_forward_validator import (
    DiagnosticEvaluator,
    FoldTrainer,
    WalkForwardFoldEngine,
    WalkForwardFoldSpec,
    WalkForwardReportGenerator,
)


# =============================================================================
# Constants
# =============================================================================

WALK_FORWARD_DIR = MODELS_DIR / "walk_forward"

NON_FEATURE_COLS = {"datetime_utc", "dt", "dominant_pollutant", "aqi_category"}

DISTRIBUTION_SHIFT_COLS = ["pm2_5", "epa_aqi", "temperature_2m", "wind_speed_10m"]

TARGET_COLS = [f"target_h{h}" for h in range(1, 73)]

# Model factories (callables that produce fresh unfitted model instances)
MODEL_FACTORIES: dict[str, Any] = {
    "exp019": lambda: PersistenceAwareHybridModel(
        short_estimators=40, short_lr=0.1, ridge_alpha=1.0, min_blend_weight=0.6
    ),
    "exp017": lambda: HybridAQISpecialistModel(
        short_estimators=40, short_lr=0.1, ridge_alpha=1.0
    ),
    "ridge_v2": lambda: RidgeAQIModel(alpha=1.0),
    "naive": lambda: NaivePersistenceBaseline(),
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "exp019": "Persistence-Aware Hybrid (EXP-019)",
    "exp017": "Hybrid AQI Specialist (EXP-017)",
    "ridge_v2": "Ridge v2 Weather-Enriched",
    "naive": "Naive Persistence Baseline",
}


# =============================================================================
# Data Loading
# =============================================================================

def load_full_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load enriched feature dataframe and full target dataframe.

    Returns:
        (df_features, df_targets): Dataframes indexed by datetime_utc.
    """
    features_path = PROCESSED_DATA_DIR / "features_v2_weather.csv"
    targets_path = PROCESSED_DATA_DIR / "y_train.csv"

    logger.info(f"Loading features from {features_path}")
    df_features = pd.read_csv(features_path, parse_dates=["datetime_utc"])
    df_features["datetime_utc"] = pd.to_datetime(df_features["datetime_utc"], utc=True)

    logger.info(f"Loading targets from {targets_path}")
    df_targets_raw = pd.read_csv(targets_path, parse_dates=["datetime_utc"])
    df_targets_raw["datetime_utc"] = pd.to_datetime(df_targets_raw["datetime_utc"], utc=True)

    # Also load the full targets (test set) to reconstruct the complete target series
    # for walk-forward folds that may span into fold validation windows.
    # NOTE: We load ALL features but will filter by fold boundaries at evaluation time.
    # y_test is NOT used for scoring — only for reconstructing target labels in val windows.
    y_test_path = PROCESSED_DATA_DIR / "y_test.csv"
    if y_test_path.exists():
        df_test_raw = pd.read_csv(y_test_path, parse_dates=["datetime_utc"])
        df_test_raw["datetime_utc"] = pd.to_datetime(df_test_raw["datetime_utc"], utc=True)
        df_targets = pd.concat([df_targets_raw, df_test_raw], ignore_index=True)
    else:
        df_targets = df_targets_raw

    df_targets = df_targets.sort_values("datetime_utc").drop_duplicates("datetime_utc")

    logger.info(
        f"Features: {df_features.shape}, "
        f"Targets: {df_targets.shape}, "
        f"Targets range: {df_targets['datetime_utc'].min()} to {df_targets['datetime_utc'].max()}"
    )
    return df_features, df_targets


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names (excluding metadata columns)."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


# =============================================================================
# Per-Fold Evaluation
# =============================================================================

def evaluate_model_on_fold(
    model_key: str,
    model_factory,
    fold: WalkForwardFoldSpec,
    df_features: pd.DataFrame,
    df_targets: pd.DataFrame,
    feature_cols: list[str],
    evaluator: DiagnosticEvaluator,
) -> dict[str, Any]:
    """Train and evaluate one model on one fold.

    Args:
        model_key: Short identifier (e.g. 'exp019').
        model_factory: Callable returning an unfitted model instance.
        fold: Fold spec with train/val boundaries.
        df_features: Full feature dataframe.
        df_targets: Full target dataframe.
        feature_cols: List of feature column names to use.
        evaluator: DiagnosticEvaluator instance.

    Returns:
        Dictionary with model metadata, fit info, and all 5 diagnostic dimensions.
    """
    logger.info(f"[{fold.fold_id}] Training {model_key} ...")

    # --- Build train / val masks ---
    train_feat_mask = df_features["datetime_utc"] <= fold.train_end
    val_feat_mask = (
        (df_features["datetime_utc"] >= fold.val_start)
        & (df_features["datetime_utc"] <= fold.val_end)
    )

    train_target_mask = df_targets["datetime_utc"] <= fold.train_end
    val_target_mask = (
        (df_targets["datetime_utc"] >= fold.val_start)
        & (df_targets["datetime_utc"] <= fold.val_end)
    )

    df_train_feat = df_features[train_feat_mask].copy()
    df_val_feat = df_features[val_feat_mask].copy()
    df_train_targets = df_targets[train_target_mask].copy()
    df_val_targets = df_targets[val_target_mask].copy()

    # Align features and targets on datetime_utc
    df_train_feat = df_train_feat[df_train_feat["datetime_utc"].isin(df_train_targets["datetime_utc"])]
    df_val_feat = df_val_feat[df_val_feat["datetime_utc"].isin(df_val_targets["datetime_utc"])]
    df_train_targets = df_train_targets[df_train_targets["datetime_utc"].isin(df_train_feat["datetime_utc"])]
    df_val_targets = df_val_targets[df_val_targets["datetime_utc"].isin(df_val_feat["datetime_utc"])]

    n_train = len(df_train_feat)
    n_val = len(df_val_feat)

    if n_val < 24:
        logger.warning(f"[{fold.fold_id}/{model_key}] Very few validation samples: {n_val}")

    # --- Extract arrays ---
    X_train_raw = df_train_feat[feature_cols].values.astype(np.float32)
    y_train = df_train_targets[TARGET_COLS].values.astype(np.float32)
    X_val_raw = df_val_feat[feature_cols].values.astype(np.float32)
    y_val = df_val_targets[TARGET_COLS].values.astype(np.float32)

    # --- Fit scaler FROM SCRATCH on fold training data only ---
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    # --- Create and fit fresh model instance ---
    model = model_factory()
    if model_key == "naive":
        # Naive uses raw current AQI, no scaling needed
        current_aqi_val = df_val_feat["epa_aqi"].values
        y_pred = model.predict_from_current(current_aqi_val)
    elif model_key == "exp019":
        model.fit(X_train, y_train)
        current_aqi_val = df_val_feat["epa_aqi"].values
        y_pred = model.predict(X_val, current_aqi=current_aqi_val)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    # --- Run diagnostics ---
    val_timestamps = df_val_feat["datetime_utc"]
    diagnostics = evaluator.evaluate(
        y_true=y_val,
        y_pred=y_pred,
        val_timestamps=val_timestamps,
        df_train_features=df_train_feat,
        df_val_features=df_val_feat,
        distribution_cols=DISTRIBUTION_SHIFT_COLS,
    )

    return {
        "model_key": model_key,
        "model_name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
        "fold_id": fold.fold_id,
        "n_train": n_train,
        "n_val": n_val,
        "train_end": str(fold.train_end),
        "val_start": str(fold.val_start),
        "val_end": str(fold.val_end),
        "diagnostics": diagnostics,
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_walk_forward_validation() -> dict[str, Any]:
    """Run 4 × 4 = 16 walk-forward evaluations and save diagnostic reports.

    Returns:
        Full results dictionary.
    """
    logger.info("=== Phase 11: Walk-Forward Stability & Diagnostic Validation ===")

    # Build and verify folds
    engine = WalkForwardFoldEngine()
    folds = engine.build_folds()
    logger.info(f"Built {len(folds)} walk-forward folds with embargo invariant verified.")

    # Load data
    df_features, df_targets = load_full_data()
    feature_cols = _get_feature_cols(df_features)
    logger.info(f"Using {len(feature_cols)} feature columns.")

    evaluator = DiagnosticEvaluator()
    results: dict[str, Any] = {
        "metadata": {
            "phase": "Phase 11: Walk-Forward Stability & Diagnostic Validation",
            "n_folds": len(folds),
            "n_models": len(MODEL_FACTORIES),
            "total_evaluations": len(folds) * len(MODEL_FACTORIES),
            "feature_count": len(feature_cols),
            "embargo_hours": 72,
            "season_coverage_note": (
                "Folds are forecast-origin windows. Seasons are per-sample labels "
                "applied at evaluation time via LahoreSeasonClassifier. "
                "Fold 1 & 4: winter_smog. Fold 2: transition + summer. Fold 3: monsoon."
            ),
        },
        "folds": {},
    }

    # 4 folds × 4 models = 16 evaluations
    for fold in folds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold: {fold.fold_id} | Val: {fold.val_start} to {fold.val_end}")
        logger.info(f"Primary seasons: {fold.primary_seasons}")
        logger.info(f"{'='*60}")

        fold_results: dict[str, Any] = {
            "fold_id": fold.fold_id,
            "train_end": str(fold.train_end),
            "val_start": str(fold.val_start),
            "val_end": str(fold.val_end),
            "embargo_hours": fold.embargo_hours,
            "primary_seasons": fold.primary_seasons,
            "models": {},
        }

        for model_key, model_factory in MODEL_FACTORIES.items():
            try:
                model_result = evaluate_model_on_fold(
                    model_key=model_key,
                    model_factory=model_factory,
                    fold=fold,
                    df_features=df_features,
                    df_targets=df_targets,
                    feature_cols=feature_cols,
                    evaluator=evaluator,
                )
                fold_results["models"][model_key] = model_result
                overall = model_result["diagnostics"]["overall"]
                logger.info(
                    f"  [{model_key}] RMSE={overall['rmse']:.2f} "
                    f"MAE={overall['mae']:.2f} R²={overall['r2']:.4f}"
                )
            except Exception as e:
                logger.error(f"  [{model_key}] FAILED: {e}")
                fold_results["models"][model_key] = {"error": str(e)}

        results["folds"][fold.fold_id] = fold_results

    # Build stability report
    reporter = WalkForwardReportGenerator(WALK_FORWARD_DIR)
    stability = reporter.build_stability_report(results)
    results["stability_report"] = stability

    # Log stability summary
    if stability:
        logger.info("\n=== EXP-019 vs Naive Stability ===")
        logger.info(f"  Fold wins (EXP-019): {stability['exp019_wins']}/{stability['fold_count']}")
        logger.info(f"  Mean RMSE delta (+ = EXP-019 wins): {stability['mean_delta']:.3f}")
        logger.info(f"  Median delta: {stability['median_delta']:.3f}")
        logger.info(f"  Std: {stability['std_delta']:.3f}")
        logger.info(f"  Worst fold delta: {stability['worst_fold_delta']:.3f}")
        logger.info(f"  Relative gain vs Naive: {stability['relative_gain_percent']:.2f}%")

    # Save reports
    json_path = reporter.save_json_report(results)
    csv_path = reporter.save_summary_csv(results)

    logger.info(f"\n=== Reports Saved ===")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  CSV: {csv_path}")
    logger.info("=== Phase 11 Complete ===")

    return results


if __name__ == "__main__":
    run_walk_forward_validation()
