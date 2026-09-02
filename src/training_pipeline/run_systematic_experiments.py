"""Pearls AQI Predictor - Systematic Experiment Suite Runner (Phase 10.5C).

Executes reproducible, chronological training-only cross-validation across:
1. Ridge alpha grid search (1e-4 ... 1e4)
2. ElasticNet vs Ridge regularizations
3. LightGBM Direct Multi-Output vs Horizon-as-Feature strategies

All runs are recorded in the ExperimentRegistry without touching the held-out test set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.multioutput import MultiOutputRegressor

from src.config import PROCESSED_DATA_DIR
from src.logger import logger
from src.models.lightgbm_models import (
    LightGBMDirectMultiOutput,
    LightGBMHorizonAsFeature,
)
from src.training_pipeline.cv_evaluator import ChronologicalCVEvaluator
from src.training_pipeline.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
)


def run_phase_10_5c_experiment_suite(
    data_dir: Path = PROCESSED_DATA_DIR,
) -> ExperimentRegistry:
    """Run full systematic model tuning suite across expanding training folds."""
    registry = ExperimentRegistry()
    cv_evaluator = ChronologicalCVEvaluator(n_folds=3, min_train_fraction=0.55, val_fraction_per_fold=0.15)

    # Load weather-enriched training dataset (v2)
    logger.info("Loading weather-enriched training dataset (v2)...")
    X_train_v2 = np.load(data_dir / "X_train_v2.npy")
    y_train_v2 = np.load(data_dir / "y_train_v2.npy")

    # =========================================================================
    # SUITE 1: Ridge Alpha Logarithmic Grid Search (EXP-001 to EXP-009)
    # =========================================================================
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    logger.info(f"=== SUITE 1: Ridge Alpha Tuning ({len(alpha_grid)} configurations) ===")

    for idx, alpha in enumerate(alpha_grid, start=1):
        exp_id = f"EXP-{idx:03d}"
        logger.info(f"Running {exp_id}: Ridge (alpha={alpha})...")

        def _make_ridge(a=alpha):
            return Ridge(alpha=a, random_state=42)

        cv_res = cv_evaluator.evaluate_model_cv(_make_ridge, X_train_v2, y_train_v2)

        record = ExperimentRecord(
            experiment_id=exp_id,
            feature_version="v2_weather_enriched",
            model_name="Ridge Regression",
            hyperparameters={"alpha": alpha},
            strategy="multioutput_direct",
            cv_scheme="expanding_3fold_train_only",
            val_rmse_mean=cv_res["val_rmse_mean"],
            val_rmse_std=cv_res["val_rmse_std"],
            val_mae_mean=cv_res["val_mae_mean"],
            val_r2_mean=cv_res["val_r2_mean"],
            val_h1_rmse=cv_res["val_h1_rmse"],
            val_h24_rmse=cv_res["val_h24_rmse"],
            val_h72_rmse=cv_res["val_h72_rmse"],
            training_time_sec=cv_res["training_time_sec"],
            notes=f"Ridge alpha={alpha} tuning on weather-enriched v2 features",
        )
        registry.log_experiment(record)

    # =========================================================================
    # SUITE 2: ElasticNet Regularization (EXP-010 to EXP-012)
    # =========================================================================
    l1_ratios = [0.1, 0.5, 0.9]
    logger.info(f"=== SUITE 2: ElasticNet Tuning ({len(l1_ratios)} configurations) ===")

    for idx, l1_ratio in enumerate(l1_ratios, start=10):
        exp_id = f"EXP-{idx:03d}"
        logger.info(f"Running {exp_id}: ElasticNet (alpha=0.1, l1_ratio={l1_ratio})...")

        def _make_elasticnet(l1=l1_ratio):
            return MultiOutputRegressor(
                ElasticNet(alpha=0.1, l1_ratio=l1, tol=0.01, max_iter=50, random_state=42),
                n_jobs=-1,
            )

        cv_res = cv_evaluator.evaluate_model_cv(_make_elasticnet, X_train_v2, y_train_v2)

        record = ExperimentRecord(
            experiment_id=exp_id,
            feature_version="v2_weather_enriched",
            model_name="ElasticNet",
            hyperparameters={"alpha": 0.1, "l1_ratio": l1_ratio},
            strategy="multioutput_direct",
            cv_scheme="expanding_3fold_train_only",
            val_rmse_mean=cv_res["val_rmse_mean"],
            val_rmse_std=cv_res["val_rmse_std"],
            val_mae_mean=cv_res["val_mae_mean"],
            val_r2_mean=cv_res["val_r2_mean"],
            val_h1_rmse=cv_res["val_h1_rmse"],
            val_h24_rmse=cv_res["val_h24_rmse"],
            val_h72_rmse=cv_res["val_h72_rmse"],
            training_time_sec=cv_res["training_time_sec"],
            notes=f"ElasticNet l1_ratio={l1_ratio} with alpha=0.1",
        )
        registry.log_experiment(record)

    # =========================================================================
    # SUITE 3: LightGBM Gradient Boosting (EXP-013 & EXP-014)
    # =========================================================================
    logger.info("=== SUITE 3: LightGBM Gradient Boosted Trees ===")

    # 3.1 LightGBM Direct Multi-Output
    logger.info("Running EXP-013: LightGBM Direct Multi-Output (72 estimators)...")
    def _make_lgbm_direct():
        return LightGBMDirectMultiOutput(n_estimators=40, learning_rate=0.1, num_leaves=20, n_jobs=-1)

    cv_res_lgbm_direct = cv_evaluator.evaluate_model_cv(_make_lgbm_direct, X_train_v2, y_train_v2)
    record_lgbm_direct = ExperimentRecord(
        experiment_id="EXP-013",
        feature_version="v2_weather_enriched",
        model_name="LightGBM Direct",
        hyperparameters={"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 20},
        strategy="direct_72_estimators",
        cv_scheme="expanding_3fold_train_only",
        val_rmse_mean=cv_res_lgbm_direct["val_rmse_mean"],
        val_rmse_std=cv_res_lgbm_direct["val_rmse_std"],
        val_mae_mean=cv_res_lgbm_direct["val_mae_mean"],
        val_r2_mean=cv_res_lgbm_direct["val_r2_mean"],
        val_h1_rmse=cv_res_lgbm_direct["val_h1_rmse"],
        val_h24_rmse=cv_res_lgbm_direct["val_h24_rmse"],
        val_h72_rmse=cv_res_lgbm_direct["val_h72_rmse"],
        training_time_sec=cv_res_lgbm_direct["training_time_sec"],
        notes="LightGBM with 72 independent horizon regressors",
    )
    registry.log_experiment(record_lgbm_direct)

    # 3.2 LightGBM Horizon as Feature
    logger.info("Running EXP-014: LightGBM Horizon as Input Feature...")
    def _make_lgbm_horizon_feat():
        return LightGBMHorizonAsFeature(n_estimators=80, learning_rate=0.1, num_leaves=31, n_jobs=-1)

    cv_res_lgbm_h = cv_evaluator.evaluate_model_cv(_make_lgbm_horizon_feat, X_train_v2, y_train_v2)
    record_lgbm_h = ExperimentRecord(
        experiment_id="EXP-014",
        feature_version="v2_weather_enriched",
        model_name="LightGBM Horizon-as-Feature",
        hyperparameters={"n_estimators": 80, "learning_rate": 0.1, "num_leaves": 31},
        strategy="horizon_as_feature",
        cv_scheme="expanding_3fold_train_only",
        val_rmse_mean=cv_res_lgbm_h["val_rmse_mean"],
        val_rmse_std=cv_res_lgbm_h["val_rmse_std"],
        val_mae_mean=cv_res_lgbm_h["val_mae_mean"],
        val_r2_mean=cv_res_lgbm_h["val_r2_mean"],
        val_h1_rmse=cv_res_lgbm_h["val_h1_rmse"],
        val_h24_rmse=cv_res_lgbm_h["val_h24_rmse"],
        val_h72_rmse=cv_res_lgbm_h["val_h72_rmse"],
        training_time_sec=cv_res_lgbm_h["training_time_sec"],
        notes="LightGBM with normalized horizon step passed as input feature",
    )
    registry.log_experiment(record_lgbm_h)

    logger.info("=== All Systematic Experiments Completed Successfully ===")
    return registry


if __name__ == "__main__":
    run_phase_10_5c_experiment_suite()
