"""Pearls AQI Predictor - Explainability Artifacts Precomputation Script.

Precomputes and stores:
1. `shap_background.npy`: 100 representative background samples.
2. `global_shap_importance.json`: Precomputed on 500 seasonal stratified samples across all 72 horizons.
3. `explainer_manifest.json`: Linked to the EXP-019 model SHA256 checksum.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.inference.explainer import ModelExplainer
from src.inference.hopsworks_registry import compute_file_sha256
from src.inference.model_loader import ModelLoader
from src.logger import logger

EXPLAINABILITY_DIR = MODELS_DIR / "explainability"


def build_artifacts(
    n_background: int = 100,
    n_cohort: int = 500,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate and save precomputed SHAP explainability artifacts.

    Args:
        n_background: Number of background reference samples (default: 100).
        n_cohort: Number of stratified samples for global importance evaluation (default: 500).
        output_dir: Output directory (default: data/models/explainability).

    Returns:
        Manifest summary dictionary.
    """
    if output_dir is None:
        output_dir = EXPLAINABILITY_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ModelLoader()
    model = loader.load_model()
    scaler = loader.load_scaler()
    features = loader.load_schema()

    # Load dataset
    csv_path = PROCESSED_DATA_DIR / "features_v2_weather.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # Ensure all canonical columns present
    df_clean = df[features].dropna().copy()
    n_total = len(df_clean)

    # 1. Deterministic background sampling (first n_background rows)
    bg_raw = df_clean.head(n_background).values
    bg_scaled = scaler.transform(bg_raw)
    bg_file = output_dir / "shap_background.npy"
    np.save(bg_file, bg_scaled)
    logger.info(f"Saved SHAP background ({n_background} samples) to {bg_file}")

    # 2. Stratified / representative cohort sampling (e.g. evenly spaced over training set)
    step = max(1, n_total // n_cohort)
    cohort_indices = np.arange(0, n_total, step)[:n_cohort]
    cohort_df = df_clean.iloc[cohort_indices]
    cohort_raw = cohort_df.values

    explainer = ModelExplainer(
        model=model,
        scaler=scaler,
        feature_names=features,
        background_data=bg_scaled,
    )

    logger.info(f"Computing per-horizon global SHAP attributions across {len(cohort_raw)} samples...")

    # We evaluate key milestones and all 72 horizons
    # Accumulate per-feature absolute SHAP sums: shape (72, 114)
    all_horizon_mean_abs = np.zeros((72, 114), dtype=np.float64)
    persistence_contrib_by_h = np.zeros(72, dtype=np.float64)

    # Current AQI vector from cohort
    current_aqis = cohort_raw[:, model.current_aqi_col_idx]

    # Pre-scale cohort for fast evaluation
    cohort_scaled = scaler.transform(cohort_raw)  # (N, 114)

    # For h1..6 (LightGBM):
    for h in range(1, 7):
        tree_exp = explainer._tree_explainers[h - 1]
        raw_shap = tree_exp.shap_values(cohort_scaled, check_additivity=False)
        if isinstance(raw_shap, list):
            raw_shap = raw_shap[0]
        # raw_shap shape: (N, 114)
        all_horizon_mean_abs[h - 1] = np.mean(np.abs(raw_shap), axis=0)

    # For h7..72 (Ridge / Blended Ridge):
    for h in range(7, 73):
        ridge_idx = h - 7
        coef = model.m_rest.coef_[ridge_idx]  # (114,)
        # Linear SHAP: coef_i * (X_scaled_i - bg_mean_i)
        phi_unweighted = (cohort_scaled - explainer.bg_means) * coef  # (N, 114)

        if h <= 37:
            w_h = 1.0
            persistence_contrib_by_h[h - 1] = 0.0
        else:
            w_h = float(explainer.blend_weights[h - 38])
            persistence_contrib_by_h[h - 1] = float(np.mean((1.0 - w_h) * np.abs(current_aqis)))

        phi_final = w_h * phi_unweighted
        all_horizon_mean_abs[h - 1] = np.mean(np.abs(phi_final), axis=0)

    # Overall global feature importance across all 72 horizons
    global_feature_importance_values = np.mean(all_horizon_mean_abs, axis=0)  # (114,)
    global_persistence_mean = float(np.mean(persistence_contrib_by_h))

    # Sort overall ranking
    sorted_indices = np.argsort(global_feature_importance_values)[::-1]
    overall_ranking = [
        {
            "rank": int(rank + 1),
            "feature": features[idx],
            "mean_abs_shap": round(float(global_feature_importance_values[idx]), 4),
        }
        for rank, idx in enumerate(sorted_indices)
    ]

    # Milestone horizon rankings
    milestones = [1, 6, 12, 24, 48, 72]
    milestone_rankings = {}
    for mh in milestones:
        mh_vals = all_horizon_mean_abs[mh - 1]
        mh_sorted = np.argsort(mh_vals)[::-1]
        milestone_rankings[f"h{mh}"] = {
            "specialist_type": "LightGBM" if mh <= 6 else ("Ridge" if mh <= 37 else "Ridge+Persistence"),
            "persistence_mean_abs": round(float(persistence_contrib_by_h[mh - 1]), 4),
            "top_10": [
                {
                    "rank": r + 1,
                    "feature": features[idx],
                    "mean_abs_shap": round(float(mh_vals[idx]), 4),
                }
                for r, idx in enumerate(mh_sorted[:10])
            ],
        }

    global_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": "EXP-019",
        "cohort_samples": len(cohort_raw),
        "background_samples": n_background,
        "global_persistence_mean_contribution": round(global_persistence_mean, 4),
        "overall_feature_importance": overall_ranking,
        "milestone_horizon_rankings": milestone_rankings,
    }

    global_file = output_dir / "global_shap_importance.json"
    with open(global_file, "w", encoding="utf-8") as f:
        json.dump(global_data, f, indent=2)
    logger.info(f"Saved global SHAP importance to {global_file}")

    # 3. Save Explainer Manifest referencing EXP-019 hash
    prod_model_path = MODELS_DIR / "production_hybrid_model.joblib"
    prod_hash = compute_file_sha256(prod_model_path) if prod_model_path.exists() else "unknown"

    manifest = {
        "model_id": "EXP-019",
        "production_model_sha256": prod_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "background_samples": n_background,
        "cohort_samples": len(cohort_raw),
        "features_count": 114,
        "output_horizons": 72,
        "artifacts": {
            "shap_background.npy": compute_file_sha256(bg_file),
            "global_shap_importance.json": compute_file_sha256(global_file),
        },
    }

    manifest_file = output_dir / "explainer_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved explainer manifest to {manifest_file}")

    return manifest


if __name__ == "__main__":
    build_artifacts()
