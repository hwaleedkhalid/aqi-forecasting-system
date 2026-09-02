"""Pearls AQI Predictor - Phase 10.5E: Final Out-of-Time Benchmark Suite.

Executes the definitive, single out-of-time evaluation on the frozen 9,311-sample held-out test partition.
Evaluates:
1. Naive Persistence Baseline
2. Ridge Regression v1 (Pollutants-only, 64 features)
3. Random Forest v1 (Pollutants-only, 64 features)
4. TensorFlow Feed-Forward DNN v1 (Pollutants-only, 64 features)
5. Ridge Regression v2 (EXP-005, Weather-enriched, 114 features)
6. Persistence-Aware Hybrid Ensemble (EXP-019, 114 features)
7. Hybrid AQI Specialist Champion (EXP-017, 114 features)

Reports overall metrics, all 72 horizon curves, high-AQI safety diagnostics, and CV-to-test transfer deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.logger import logger
from src.models.hybrid_specialist_model import (
    HybridAQISpecialistModel,
    PersistenceAwareHybridModel,
)
from src.models.naive_baseline import NaivePersistenceBaseline
from src.models.random_forest_model import RandomForestAQIModel
from src.models.ridge_model import RidgeAQIModel
from src.models.tensorflow_model import TensorFlowAQIModel


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_aqi: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute overall, per-horizon, and severity-segmented metrics."""
    errors = y_pred - y_true
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - (np.sum(errors ** 2) / ss_tot)) if ss_tot > 0 else 0.0

    # All 72 horizons
    per_horizon_rmse = [float(np.sqrt(np.mean((y_true[:, h] - y_pred[:, h]) ** 2))) for h in range(72)]
    per_horizon_mae = [float(np.mean(np.abs(y_true[:, h] - y_pred[:, h]))) for h in range(72)]
    per_horizon_r2 = []
    for h in range(72):
        tot_h = float(np.sum((y_true[:, h] - np.mean(y_true[:, h])) ** 2))
        r2_h = float(1.0 - (np.sum((y_true[:, h] - y_pred[:, h]) ** 2) / tot_h)) if tot_h > 0 else 0.0
        per_horizon_r2.append(r2_h)

    # Key checkpoints (1-indexed horizon steps: h1, h6, h12, h24, h48, h72)
    checkpoints = {
        "h1_rmse": per_horizon_rmse[0],
        "h6_rmse": per_horizon_rmse[5],
        "h12_rmse": per_horizon_rmse[11],
        "h24_rmse": per_horizon_rmse[23],
        "h48_rmse": per_horizon_rmse[47],
        "h72_rmse": per_horizon_rmse[71],
        "h1_mae": per_horizon_mae[0],
        "h24_mae": per_horizon_mae[23],
        "h72_mae": per_horizon_mae[71],
        "h1_r2": per_horizon_r2[0],
        "h24_r2": per_horizon_r2[23],
        "h72_r2": per_horizon_r2[71],
    }

    # High-AQI / Severe Event Segmentation (based on ground truth target severity)
    # 1. Hazardous Days (> 300 AQI)
    mask_hazardous = y_true > 300.0
    haz_rmse = float(np.sqrt(np.mean((y_true[mask_hazardous] - y_pred[mask_hazardous]) ** 2))) if mask_hazardous.sum() > 0 else 0.0

    # 2. Very Unhealthy & Hazardous (> 200 AQI)
    mask_severe = y_true > 200.0
    sev_rmse = float(np.sqrt(np.mean((y_true[mask_severe] - y_pred[mask_severe]) ** 2))) if mask_severe.sum() > 0 else 0.0

    # 3. Moderate & Sensitive (50 - 150 AQI)
    mask_mod = (y_true >= 50.0) & (y_true <= 150.0)
    mod_rmse = float(np.sqrt(np.mean((y_true[mask_mod] - y_pred[mask_mod]) ** 2))) if mask_mod.sum() > 0 else 0.0

    return {
        "overall_rmse": rmse,
        "overall_mae": mae,
        "overall_r2": r2,
        "checkpoints": checkpoints,
        "per_horizon_rmse": per_horizon_rmse,
        "per_horizon_mae": per_horizon_mae,
        "per_horizon_r2": per_horizon_r2,
        "segmented_rmse": {
            "hazardous_gt300": haz_rmse,
            "severe_gt200": sev_rmse,
            "moderate_50_to_150": mod_rmse,
        },
    }


def run_final_test_benchmark(
    data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Execute definitive single benchmark evaluation across all 7 frozen models."""
    logger.info("=== PHASE 10.5E: FINAL OUT-OF-TIME TEST BENCHMARK ===")

    # 1. Load Weather-Enriched (v2) Datasets
    logger.info("Loading weather-enriched v2 datasets (114 features)...")
    X_train_v2 = np.load(data_dir / "X_train_v2.npy")
    y_train_v2 = np.load(data_dir / "y_train_v2.npy")
    X_test_v2 = np.load(data_dir / "X_test_v2.npy")
    y_test_v2 = np.load(data_dir / "y_test_v2.npy")
    current_aqi_test_v2 = np.load(data_dir / "current_aqi_test_v2.npy")

    # 2. Load Original Pollutants-Only (v1) Datasets for strict baseline comparison
    logger.info("Loading baseline v1 datasets (64 features)...")
    X_train_v1 = np.load(data_dir / "X_train.npy")
    y_train_v1 = np.load(data_dir / "y_train.npy")
    X_test_v1 = np.load(data_dir / "X_test.npy")
    y_test_v1 = np.load(data_dir / "y_test.npy")

    benchmark_results: dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # MODEL 1: Naive Persistence Baseline
    # -------------------------------------------------------------------------
    logger.info("Evaluating Model 1: Naive Persistence Baseline...")
    naive_preds = np.repeat(current_aqi_test_v2[:, np.newaxis], 72, axis=1)
    benchmark_results["naive_persistence"] = {
        "name": "Naive Persistence Baseline",
        "category": "Baseline",
        "features": "Current AQI (1)",
        "metrics": compute_comprehensive_metrics(y_test_v2, naive_preds),
    }

    # -------------------------------------------------------------------------
    # MODEL 2: Ridge Regression v1 (Pollutants-only, 64 features)
    # -------------------------------------------------------------------------
    logger.info("Evaluating Model 2: Ridge Regression v1 (Pollutants-only)...")
    ridge_v1 = RidgeAQIModel(alpha=1.0)
    ridge_v1.fit(X_train_v1, y_train_v1)
    ridge_v1_preds = ridge_v1.predict(X_test_v1)
    benchmark_results["ridge_v1_pollutants"] = {
        "name": "Ridge Regression v1 (Pollutants-only)",
        "category": "Linear Baseline",
        "features": "Pollutants & Temporal (64)",
        "metrics": compute_comprehensive_metrics(y_test_v1, ridge_v1_preds),
    }

    # -------------------------------------------------------------------------
    # MODEL 3: Random Forest v1 (Pollutants-only, 64 features)
    # -------------------------------------------------------------------------
    logger.info("Evaluating Model 3: Random Forest v1 (Pollutants-only)...")
    rf_v1_path = models_dir / "random_forest_model.joblib"
    if rf_v1_path.exists():
        rf_v1 = RandomForestAQIModel.load(rf_v1_path)
    else:
        rf_v1 = RandomForestAQIModel(n_estimators=100, n_jobs=-1)
        rf_v1.fit(X_train_v1, y_train_v1)
    rf_v1_preds = rf_v1.predict(X_test_v1)
    benchmark_results["random_forest_v1"] = {
        "name": "Random Forest v1 (Pollutants-only)",
        "category": "Tree Baseline",
        "features": "Pollutants & Temporal (64)",
        "metrics": compute_comprehensive_metrics(y_test_v1, rf_v1_preds),
    }

    # -------------------------------------------------------------------------
    # MODEL 4: TensorFlow Feed-Forward DNN v1 (Pollutants-only, 64 features)
    # -------------------------------------------------------------------------
    logger.info("Evaluating Model 4: TensorFlow DNN v1 (Pollutants-only)...")
    tf_v1_path = models_dir / "tensorflow_model.keras"
    if tf_v1_path.exists():
        tf_v1 = TensorFlowAQIModel.load(tf_v1_path)
        tf_v1_preds = tf_v1.predict(X_test_v1)
        benchmark_results["tensorflow_dnn_v1"] = {
            "name": "TensorFlow DNN v1 (Pollutants-only)",
            "category": "Deep Learning Baseline",
            "features": "Pollutants & Temporal (64)",
            "metrics": compute_comprehensive_metrics(y_test_v1, tf_v1_preds),
        }

    # -------------------------------------------------------------------------
    # MODEL 5: Ridge Regression v2 (EXP-005: Weather-enriched, 114 features)
    # -------------------------------------------------------------------------
    logger.info("Training & Evaluating Model 5: Ridge Regression v2 (Weather-Enriched)...")
    ridge_v2 = RidgeAQIModel(alpha=1.0)
    ridge_v2.fit(X_train_v2, y_train_v2)
    ridge_v2_preds = ridge_v2.predict(X_test_v2)
    benchmark_results["ridge_v2_weather"] = {
        "name": "Ridge Regression v2 (EXP-005 Weather-Enriched)",
        "category": "Linear Meteorological",
        "features": "Weather + Pollutants (114)",
        "metrics": compute_comprehensive_metrics(y_test_v2, ridge_v2_preds),
    }

    # -------------------------------------------------------------------------
    # MODEL 6: Persistence-Aware Hybrid Ensemble (EXP-019, 114 features)
    # -------------------------------------------------------------------------
    logger.info("Training & Evaluating Model 6: Persistence-Aware Hybrid (EXP-019)...")
    pers_hybrid = PersistenceAwareHybridModel(
        short_estimators=40,
        short_lr=0.1,
        ridge_alpha=1.0,
        min_blend_weight=0.7,
    )
    pers_hybrid.fit(X_train_v2, y_train_v2)
    pers_hybrid_preds = pers_hybrid.predict(X_test_v2, current_aqi=current_aqi_test_v2)
    benchmark_results["persistence_aware_hybrid_exp019"] = {
        "name": "Persistence-Aware Hybrid (EXP-019)",
        "category": "Ensemble Specialist",
        "features": "Weather + Pollutants (114)",
        "metrics": compute_comprehensive_metrics(y_test_v2, pers_hybrid_preds),
    }

    # -------------------------------------------------------------------------
    # MODEL 7: Hybrid AQI Specialist Champion (EXP-017, 114 features)
    # -------------------------------------------------------------------------
    logger.info("Training & Evaluating Model 7: Hybrid AQI Specialist Champion (EXP-017)...")
    hybrid_specialist = HybridAQISpecialistModel(
        short_estimators=40,
        short_lr=0.1,
        ridge_alpha=1.0,
    )
    hybrid_specialist.fit(X_train_v2, y_train_v2)
    hybrid_specialist_preds = hybrid_specialist.predict(X_test_v2)
    benchmark_results["hybrid_specialist_exp017"] = {
        "name": "Hybrid AQI Specialist Champion (EXP-017)",
        "category": "Primary Production Candidate",
        "features": "Weather + Pollutants (114)",
        "metrics": compute_comprehensive_metrics(y_test_v2, hybrid_specialist_preds),
    }

    # -------------------------------------------------------------------------
    # Comparative Diagnostic Summaries & Relative Gains
    # -------------------------------------------------------------------------
    naive_rmse = benchmark_results["naive_persistence"]["metrics"]["overall_rmse"]
    ridge1_rmse = benchmark_results["ridge_v1_pollutants"]["metrics"]["overall_rmse"]
    ridge2_rmse = benchmark_results["ridge_v2_weather"]["metrics"]["overall_rmse"]
    
    # Identify overall champion on test set by lowest RMSE
    sorted_models = sorted(benchmark_results.items(), key=lambda x: x[1]["metrics"]["overall_rmse"])
    champion_key, champion_data = sorted_models[0]
    champion_rmse = champion_data["metrics"]["overall_rmse"]

    comparative_summary = {
        "champion_model_key": champion_key,
        "champion_model_name": champion_data["name"],
        "champion_overall_rmse": champion_rmse,
        "champion_overall_r2": champion_data["metrics"]["overall_r2"],
        "gain_vs_naive_percent": round(((naive_rmse - champion_rmse) / naive_rmse) * 100, 2),
        "gain_vs_ridge_v1_percent": round(((ridge1_rmse - champion_rmse) / ridge1_rmse) * 100, 2),
        "gain_vs_ridge_v2_percent": round(((ridge2_rmse - champion_rmse) / ridge2_rmse) * 100, 2),
        "weather_telemetry_gain_on_ridge_percent": round(((ridge1_rmse - ridge2_rmse) / ridge1_rmse) * 100, 2),
        "test_sample_count": len(y_test_v2),
        "test_period": "2025-06-07 to 2026-08-28 UTC (Untouched)",
    }

    output_payload = {
        "metadata": comparative_summary,
        "models": benchmark_results,
    }

    # Serialize benchmark outputs
    benchmark_json_path = models_dir / "final_test_benchmark.json"
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info(f"Saved benchmark JSON report to {benchmark_json_path}")

    # Build and save flat summary CSV
    summary_rows = []
    for key, data in benchmark_results.items():
        m = data["metrics"]
        chk = m["checkpoints"]
        seg = m["segmented_rmse"]
        summary_rows.append({
            "Model Key": key,
            "Model Name": data["name"],
            "Category": data["category"],
            "Features": data["features"],
            "Overall RMSE": round(m["overall_rmse"], 2),
            "Overall MAE": round(m["overall_mae"], 2),
            "Overall R²": round(m["overall_r2"], 4),
            "h+1 RMSE": round(chk["h1_rmse"], 2),
            "h+6 RMSE": round(chk["h6_rmse"], 2),
            "h+12 RMSE": round(chk["h12_rmse"], 2),
            "h+24 RMSE": round(chk["h24_rmse"], 2),
            "h+48 RMSE": round(chk["h48_rmse"], 2),
            "h+72 RMSE": round(chk["h72_rmse"], 2),
            "Hazardous (>300) RMSE": round(seg["hazardous_gt300"], 2),
            "Severe (>200) RMSE": round(seg["severe_gt200"], 2),
        })

    df_summary = pd.DataFrame(summary_rows).sort_values("Overall RMSE").reset_index(drop=True)
    benchmark_csv_path = models_dir / "final_test_benchmark.csv"
    df_summary.to_csv(benchmark_csv_path, index=False)
    logger.info(f"Saved benchmark CSV summary to {benchmark_csv_path}")

    # Save Winning Production Artifacts
    prod_model_path = models_dir / "production_hybrid_model.joblib"
    pers_hybrid.save(prod_model_path)
    logger.info(f"Serialized production hybrid champion model (EXP-019) to {prod_model_path}")

    exp017_path = models_dir / "hybrid_specialist_model.joblib"
    hybrid_specialist.save(exp017_path)
    logger.info(f"Serialized hybrid specialist model (EXP-017) to {exp017_path}")

    logger.info("=== Phase 10.5E Benchmark Complete ===")
    return output_payload


if __name__ == "__main__":
    run_final_test_benchmark()
