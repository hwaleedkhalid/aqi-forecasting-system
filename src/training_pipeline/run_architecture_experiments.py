"""Pearls AQI Predictor - Phase 10.5D Forecasting Architecture Experiments Runner.

Executes 3-fold expanding chronological training-only cross-validation across:
- EXP-015: Multi-Pollutant (6 Criteria Pollutants) -> Vectorized EPA AQI Conversion
- EXP-016: Grouped-Horizon Ridge Regressor (Independent h1-6, h7-24, h25-72 models)
- EXP-017: Hybrid AQI Specialist (LightGBM h1-6 + Ridge h7-72)
- EXP-018: Hybrid Multi-Pollutant Specialist (LightGBM h1-6 + Ridge h7-72) -> EPA AQI
- EXP-019: Persistence-Aware Hybrid Ensemble (LightGBM h1-6 + Ridge h7-37 + Blended Ridge/Persistence h38-72)

All runs are recorded in the ExperimentRegistry with ZERO test set leakage.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import FORECAST_HORIZONS, PROCESSED_DATA_DIR, TRAIN_TEST_SPLIT_RATIO
from src.logger import logger
from src.models.grouped_horizon_model import GroupedHorizonModel
from src.models.hybrid_specialist_model import (
    HybridAQISpecialistModel,
    PersistenceAwareHybridModel,
)
from src.models.pollutant_to_aqi_model import (
    CRITERIA_POLLUTANTS,
    MultiPollutantToAQIModel,
)
from src.training_pipeline.cv_evaluator import ChronologicalCVEvaluator
from src.training_pipeline.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
)


def build_pollutant_target_matrices(
    data_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, np.ndarray]:
    """Build multi-horizon target matrices for each criteria pollutant matching X_train_v2 exactly.

    Returns:
        Dictionary mapping each criteria pollutant -> (37173, 72) array.
    """
    logger.info("Constructing continuous multi-pollutant training targets...")
    df_clean = pd.read_csv(data_dir / "historical_aqi_clean.csv")
    df_clean["datetime_utc"] = pd.to_datetime(df_clean["datetime_utc"], utc=True)
    df_clean = df_clean.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"]).reset_index(drop=True)

    # Reindex to continuous 1h grid
    full_grid = pd.DataFrame({
        "datetime_utc": pd.date_range(
            df_clean["datetime_utc"].min(), df_clean["datetime_utc"].max(), freq="1h", tz="UTC"
        )
    })
    
    pollutant_cols = CRITERIA_POLLUTANTS + ["epa_aqi"]
    df_grid = pd.merge(full_grid, df_clean[["datetime_utc"] + pollutant_cols], on="datetime_utc", how="left")

    # Shift each pollutant for 72 horizons (using a dict of Series for speed)
    shift_dict = {"datetime_utc": df_grid["datetime_utc"]}
    for pol in CRITERIA_POLLUTANTS:
        for h in range(1, FORECAST_HORIZONS + 1):
            shift_dict[f"target_{pol}_h{h}"] = df_grid[pol].shift(-h)

    for h in range(1, FORECAST_HORIZONS + 1):
        shift_dict[f"target_h{h}"] = df_grid["epa_aqi"].shift(-h)

    df_grid_shifted = pd.DataFrame(shift_dict)

    # Load enriched feature dataset
    df_enriched = pd.read_csv(data_dir / "features_v2_weather.csv")
    df_enriched["datetime_utc"] = pd.to_datetime(df_enriched["datetime_utc"], utc=True)

    # Merge
    merged = pd.merge(df_enriched, df_grid_shifted, on="datetime_utc", how="inner")
    merged = merged.dropna().reset_index(drop=True)

    timestamps = merged["datetime_utc"]
    total_pairs = len(merged)

    # Apply 80/20 chronological train split with 72h embargo
    split_index = int(total_pairs * TRAIN_TEST_SPLIT_RATIO)
    split_ts = timestamps.iloc[split_index]
    embargo_cutoff = split_ts - pd.Timedelta(hours=FORECAST_HORIZONS)
    train_mask = timestamps <= embargo_cutoff

    train_pollutant_targets: dict[str, np.ndarray] = {}
    for pol in CRITERIA_POLLUTANTS:
        cols = [f"target_{pol}_h{h}" for h in range(1, FORECAST_HORIZONS + 1)]
        train_pollutant_targets[pol] = merged.loc[train_mask, cols].values.astype(np.float32)

    logger.info(f"Built multi-pollutant training targets: {train_pollutant_targets['pm2_5'].shape} rows.")
    return train_pollutant_targets


def run_phase_10_5d_experiments(
    data_dir: Path = PROCESSED_DATA_DIR,
) -> ExperimentRegistry:
    """Run full suite of forecasting architecture experiments."""
    registry = ExperimentRegistry()
    cv_evaluator = ChronologicalCVEvaluator(n_folds=3, min_train_fraction=0.55, val_fraction_per_fold=0.15)

    # Load weather-enriched training dataset (v2)
    logger.info("Loading X_train_v2 and y_train_v2...")
    X_train_v2 = np.load(data_dir / "X_train_v2.npy")
    y_train_v2 = np.load(data_dir / "y_train_v2.npy")
    current_aqi_train = np.load(data_dir / "current_aqi_train_v2.npy")
    y_pollutants_train = build_pollutant_target_matrices(data_dir)

    folds = cv_evaluator.get_folds(len(X_train_v2))

    # =========================================================================
    # EXP-017: Hybrid AQI Specialist (LightGBM h1..6 + Ridge h7..72)
    # =========================================================================
    logger.info("=== Running EXP-017: Hybrid AQI Specialist (LightGBM h1..6 + Ridge h7..72) ===")
    def _make_hybrid_specialist():
        return HybridAQISpecialistModel(short_estimators=40, short_lr=0.1, ridge_alpha=1.0)

    cv_res_17 = cv_evaluator.evaluate_model_cv(_make_hybrid_specialist, X_train_v2, y_train_v2)
    record_17 = ExperimentRecord(
        experiment_id="EXP-017",
        feature_version="v2_weather_enriched",
        model_name="Hybrid Specialist",
        hyperparameters={"short_model": "LightGBM(40)", "rest_model": "Ridge(1.0)", "split_horizon": 6},
        strategy="hybrid_short_lgbm_rest_ridge",
        cv_scheme="expanding_3fold_train_only",
        val_rmse_mean=cv_res_17["val_rmse_mean"],
        val_rmse_std=cv_res_17["val_rmse_std"],
        val_mae_mean=cv_res_17["val_mae_mean"],
        val_r2_mean=cv_res_17["val_r2_mean"],
        val_h1_rmse=cv_res_17["val_h1_rmse"],
        val_h24_rmse=cv_res_17["val_h24_rmse"],
        val_h72_rmse=cv_res_17["val_h72_rmse"],
        training_time_sec=cv_res_17["training_time_sec"],
        notes="LightGBM Direct for h1..6 (short) + Ridge for h7..72 (medium/long)",
    )
    registry.log_experiment(record_17)

    # =========================================================================
    # EXP-016: Grouped-Horizon Ridge Regressors (h1-6, h7-24, h25-72)
    # =========================================================================
    logger.info("=== Running EXP-016: Grouped Horizon Ridge Regressors ===")
    def _make_grouped_ridge():
        return GroupedHorizonModel(
            h1_6_estimator=Ridge(alpha=1.0, random_state=42),
            h7_24_estimator=Ridge(alpha=1.0, random_state=42),
            h25_72_estimator=Ridge(alpha=1.0, random_state=42),
        )

    cv_res_16 = cv_evaluator.evaluate_model_cv(_make_grouped_ridge, X_train_v2, y_train_v2)
    record_16 = ExperimentRecord(
        experiment_id="EXP-016",
        feature_version="v2_weather_enriched",
        model_name="Grouped Horizon Ridge",
        hyperparameters={"groups": ["h1-6", "h7-24", "h25-72"], "alpha": 1.0},
        strategy="grouped_horizon_ridge",
        cv_scheme="expanding_3fold_train_only",
        val_rmse_mean=cv_res_16["val_rmse_mean"],
        val_rmse_std=cv_res_16["val_rmse_std"],
        val_mae_mean=cv_res_16["val_mae_mean"],
        val_r2_mean=cv_res_16["val_r2_mean"],
        val_h1_rmse=cv_res_16["val_h1_rmse"],
        val_h24_rmse=cv_res_16["val_h24_rmse"],
        val_h72_rmse=cv_res_16["val_h72_rmse"],
        training_time_sec=cv_res_16["training_time_sec"],
        notes="Three independently trained Ridge multi-output groups",
    )
    registry.log_experiment(record_16)

    # =========================================================================
    # EXP-019: Persistence-Aware Hybrid Ensemble
    # =========================================================================
    logger.info("=== Running EXP-019: Persistence-Aware Hybrid Ensemble ===")
    def _make_persistence_aware():
        return PersistenceAwareHybridModel(
            short_estimators=40,
            short_lr=0.1,
            ridge_alpha=1.0,
            min_blend_weight=0.7,
        )

    rmse_list, mae_list, r2_list = [], [], []
    h1_list, h24_list, h72_list = [], [], []
    t0 = time.time()

    for train_idx, val_idx in folds:
        X_tr, y_tr = X_train_v2[train_idx], y_train_v2[train_idx]
        X_va, y_va = X_train_v2[val_idx], y_train_v2[val_idx]
        curr_va = current_aqi_train[val_idx]

        model = _make_persistence_aware()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va, current_aqi=curr_va)

        mse = np.mean((y_va - preds) ** 2)
        mae = np.mean(np.abs(y_va - preds))
        ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
        r2 = 1.0 - (np.sum((y_va - preds) ** 2) / ss_tot) if ss_tot > 0 else 0.0

        rmse_list.append(float(np.sqrt(mse)))
        mae_list.append(float(mae))
        r2_list.append(float(r2))

        h1_list.append(float(np.sqrt(np.mean((y_va[:, 0] - preds[:, 0]) ** 2))))
        h24_list.append(float(np.sqrt(np.mean((y_va[:, 23] - preds[:, 23]) ** 2))))
        h72_list.append(float(np.sqrt(np.mean((y_va[:, 71] - preds[:, 71]) ** 2))))

    t_elapsed = time.time() - t0
    record_19 = ExperimentRecord(
        experiment_id="EXP-019",
        feature_version="v2_weather_enriched",
        model_name="Persistence-Aware Hybrid",
        hyperparameters={"short": "LGBM(40)", "med": "Ridge(1.0)", "long": "Ridge+Persistence(0.7)"},
        strategy="persistence_aware_hybrid",
        cv_scheme="expanding_3fold_train_only",
        val_rmse_mean=float(np.mean(rmse_list)),
        val_rmse_std=float(np.std(rmse_list)),
        val_mae_mean=float(np.mean(mae_list)),
        val_r2_mean=float(np.mean(r2_list)),
        val_h1_rmse=float(np.mean(h1_list)),
        val_h24_rmse=float(np.mean(h24_list)),
        val_h72_rmse=float(np.mean(h72_list)),
        training_time_sec=t_elapsed,
        notes="LightGBM (h1-6) + Ridge (h7-37) + Blended Ridge/Persistence (h38-72)",
    )
    registry.log_experiment(record_19)

    # =========================================================================
    # EXP-015: Multi-Pollutant Ridge -> Vectorized EPA AQI
    # =========================================================================
    logger.info("=== Running EXP-015: Multi-Pollutant Ridge -> Vectorized EPA AQI ===")
    t0 = time.time()
    rmse_list, mae_list, r2_list = [], [], []
    h1_list, h24_list, h72_list = [], [], []

    for train_idx, val_idx in folds:
        X_tr = X_train_v2[train_idx]
        y_dict_tr = {pol: y_pollutants_train[pol][train_idx] for pol in CRITERIA_POLLUTANTS}
        X_va, y_va = X_train_v2[val_idx], y_train_v2[val_idx]

        model_pol = MultiPollutantToAQIModel(estimator_factory=lambda: Ridge(alpha=1.0, random_state=42))
        model_pol.fit(X_tr, y_dict_tr)
        preds = model_pol.predict(X_va)

        mse = np.mean((y_va - preds) ** 2)
        mae = np.mean(np.abs(y_va - preds))
        ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
        r2 = 1.0 - (np.sum((y_va - preds) ** 2) / ss_tot) if ss_tot > 0 else 0.0

        rmse_list.append(float(np.sqrt(mse)))
        mae_list.append(float(mae))
        r2_list.append(float(r2))

        h1_list.append(float(np.sqrt(np.mean((y_va[:, 0] - preds[:, 0]) ** 2))))
        h24_list.append(float(np.sqrt(np.mean((y_va[:, 23] - preds[:, 23]) ** 2))))
        h72_list.append(float(np.sqrt(np.mean((y_va[:, 71] - preds[:, 71]) ** 2))))

    t_elapsed = time.time() - t0
    record_15 = ExperimentRecord(
        experiment_id="EXP-015",
        feature_version="v2_weather_enriched",
        model_name="Multi-Pollutant Ridge -> EPA AQI",
        hyperparameters={"pollutant_models": "6x Ridge(1.0)", "pollutants": CRITERIA_POLLUTANTS},
        strategy="multipollutant_to_epa_aqi",
        cv_scheme="expanding_3fold_train_only",
        val_rmse_mean=float(np.mean(rmse_list)),
        val_rmse_std=float(np.std(rmse_list)),
        val_mae_mean=float(np.mean(mae_list)),
        val_r2_mean=float(np.mean(r2_list)),
        val_h1_rmse=float(np.mean(h1_list)),
        val_h24_rmse=float(np.mean(h24_list)),
        val_h72_rmse=float(np.mean(h72_list)),
        training_time_sec=t_elapsed,
        notes="Predicts 6 criteria pollutants with Ridge -> converts via max EPA sub-indices",
    )
    registry.log_experiment(record_15)

    # =========================================================================
    # EXP-018: Hybrid Multi-Pollutant Specialist -> EPA AQI
    # =========================================================================
    logger.info("=== Running EXP-018: Hybrid Multi-Pollutant Specialist -> EPA AQI ===")
    t0 = time.time()
    rmse_list, mae_list, r2_list = [], [], []
    h1_list, h24_list, h72_list = [], [], []

    for train_idx, val_idx in folds:
        X_tr = X_train_v2[train_idx]
        y_dict_tr = {pol: y_pollutants_train[pol][train_idx] for pol in CRITERIA_POLLUTANTS}
        X_va, y_va = X_train_v2[val_idx], y_train_v2[val_idx]

        # For each pollutant, use HybridAQISpecialistModel
        hybrid_pollutant_models = {
            pol: HybridAQISpecialistModel(short_estimators=25, short_lr=0.1, ridge_alpha=1.0)
            for pol in CRITERIA_POLLUTANTS
        }
        model_hybrid_pol = MultiPollutantToAQIModel(pollutant_models=hybrid_pollutant_models)
        model_hybrid_pol.fit(X_tr, y_dict_tr)
        preds = model_hybrid_pol.predict(X_va)

        mse = np.mean((y_va - preds) ** 2)
        mae = np.mean(np.abs(y_va - preds))
        ss_tot = np.sum((y_va - np.mean(y_va)) ** 2)
        r2 = 1.0 - (np.sum((y_va - preds) ** 2) / ss_tot) if ss_tot > 0 else 0.0

        rmse_list.append(float(np.sqrt(mse)))
        mae_list.append(float(mae))
        r2_list.append(float(r2))

        h1_list.append(float(np.sqrt(np.mean((y_va[:, 0] - preds[:, 0]) ** 2))))
        h24_list.append(float(np.sqrt(np.mean((y_va[:, 23] - preds[:, 23]) ** 2))))
        h72_list.append(float(np.sqrt(np.mean((y_va[:, 71] - preds[:, 71]) ** 2))))

    t_elapsed = time.time() - t0
    record_18 = ExperimentRecord(
        experiment_id="EXP-018",
        feature_version="v2_weather_enriched",
        model_name="Hybrid Multi-Pollutant -> EPA AQI",
        hyperparameters={"short": "6x LGBM(25)", "rest": "6x Ridge(1.0)", "split_horizon": 6},
        strategy="hybrid_multipollutant_to_epa_aqi",
        cv_scheme="expanding_3fold_train_only",
        val_rmse_mean=float(np.mean(rmse_list)),
        val_rmse_std=float(np.std(rmse_list)),
        val_mae_mean=float(np.mean(mae_list)),
        val_r2_mean=float(np.mean(r2_list)),
        val_h1_rmse=float(np.mean(h1_list)),
        val_h24_rmse=float(np.mean(h24_list)),
        val_h72_rmse=float(np.mean(h72_list)),
        training_time_sec=t_elapsed,
        notes="Hybrid LightGBM(h1-6)+Ridge(h7-72) on 6 pollutants -> EPA AQI conversion",
    )
    registry.log_experiment(record_18)

    logger.info("=== All Phase 10.5D Architecture Experiments Completed Successfully ===")
    return registry


if __name__ == "__main__":
    run_phase_10_5d_experiments()
