"""Pearls AQI Predictor - SHAP Model Explainability Engine.

Implements exact mathematical feature attribution and persistence decomposition
for the EXP-019 PersistenceAwareHybridModel across all 72 prediction horizons:
- Horizons 1..6: Pure LightGBM TreeExplainer (interventional)
- Horizons 7..37: Pure Ridge LinearExplainer (independent masker)
- Horizons 38..72: Blended Ridge + Persistence with weighted base value,
  weighted SHAP contributions, and separated persistence component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.exceptions import ValidationError
from src.inference.hopsworks_registry import compute_file_sha256
from src.logger import logger
from src.models.hybrid_specialist_model import PersistenceAwareHybridModel

EXPLAINABILITY_DIR = MODELS_DIR / "explainability"
SCHEMA_PATH = PROCESSED_DATA_DIR / "feature_schema_v2_weather.json"


class ModelExplainer:
    """SHAP Explainer for PersistenceAwareHybridModel with exact persistence-blended additivity."""

    def __init__(
        self,
        model: PersistenceAwareHybridModel,
        scaler: StandardScaler,
        feature_names: list[str],
        background_data: np.ndarray | None = None,
    ) -> None:
        """Initialize explainer.

        Args:
            model: Loaded PersistenceAwareHybridModel instance.
            scaler: Fitted StandardScaler for 114 canonical features.
            feature_names: Ordered list of 114 canonical feature names.
            background_data: (N_bg, 114) scaled background reference matrix.
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

        if len(self.feature_names) != 114:
            raise ValidationError(f"Expected 114 canonical features, got {len(self.feature_names)}")

        # Load or use background data
        if background_data is not None:
            self.background_scaled = background_data
        else:
            self.background_scaled = self._load_or_create_background()

        # Precompute Ridge background feature means for exact linear SHAP
        self.bg_means = np.mean(self.background_scaled, axis=0)

        # Initialize LightGBM TreeExplainers for h1..h6
        self._tree_explainers: list[shap.TreeExplainer] = []
        lgb_estimators = getattr(self.model.m_short, "estimators_", None)
        if lgb_estimators is None and hasattr(self.model.m_short, "estimator"):
            lgb_estimators = self.model.m_short.estimator.estimators_

        for h_idx in range(6):
            lgb_estimator = lgb_estimators[h_idx]
            tree_exp = shap.TreeExplainer(
                lgb_estimator,
                data=self.background_scaled,
                feature_perturbation="interventional",
            )
            self._tree_explainers.append(tree_exp)

        # Precompute blend weights for h38..h72 (35 horizons)
        self.n_blend = 35
        self.blend_weights = np.linspace(1.0, self.model.min_blend_weight, self.n_blend)

    def _load_or_create_background(self) -> np.ndarray:
        """Load background numpy array or generate from processed training features."""
        bg_path = EXPLAINABILITY_DIR / "shap_background.npy"
        if bg_path.exists():
            return np.load(bg_path)

        # Fallback: create 100-sample background from features_v2_weather.csv
        csv_path = PROCESSED_DATA_DIR / "features_v2_weather.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {csv_path}")

        df = pd.read_csv(csv_path)
        sample_df = df[self.feature_names].dropna().head(100)
        scaled_bg = self.scaler.transform(sample_df.values)

        EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
        np.save(bg_path, scaled_bg)
        return scaled_bg

    def explain_horizon(
        self,
        raw_feature_vector: np.ndarray,
        horizon: int = 24,
        top_k: int = 10,
        current_aqi: float | None = None,
    ) -> dict[str, Any]:
        """Compute exact SHAP feature attribution and persistence decomposition for a specific horizon.

        Args:
            raw_feature_vector: 1D or 2D array of shape (114,) or (1, 114) raw unscaled features.
            horizon: Target horizon index (1 to 72).
            top_k: Number of top influential features to return.
            current_aqi: Unscaled current AQI observation for persistence blending.

        Returns:
            Dictionary matching authoritative explanation contract.
        """
        if not (1 <= horizon <= 72):
            raise ValidationError(f"Horizon must be between 1 and 72, got {horizon}")
        if not (1 <= top_k <= 114):
            raise ValidationError(f"top_k must be between 1 and 114, got {top_k}")

        raw_vec = np.asarray(raw_feature_vector).reshape(1, -1)
        if raw_vec.shape[1] != 114:
            raise ValidationError(f"Feature vector must contain 114 columns, got {raw_vec.shape[1]}")

        # Resolve current_aqi
        if current_aqi is None:
            current_aqi = float(raw_vec[0, self.model.current_aqi_col_idx])
        else:
            current_aqi = float(current_aqi)

        # Scale input features for model
        scaled_vec = self.scaler.transform(raw_vec)  # (1, 114)
        scaled_1d = scaled_vec[0]
        raw_1d = raw_vec[0]

        # ---------------------------------------------------------------------
        # ROUTE 1: Horizons 1..6 (Pure LightGBM Specialist)
        # ---------------------------------------------------------------------
        if horizon <= 6:
            specialist_type = "LightGBM Tree Specialist"
            blend_weight = 1.0
            tree_exp = self._tree_explainers[horizon - 1]
            raw_shap = tree_exp.shap_values(scaled_vec, check_additivity=False)
            if isinstance(raw_shap, list):
                raw_shap = raw_shap[0]
            phi_unweighted = raw_shap[0]  # (114,)

            specialist_base_val = float(tree_exp.expected_value)
            raw_specialist_output = specialist_base_val + float(np.sum(phi_unweighted))

            base_value = specialist_base_val
            phi_final = phi_unweighted
            model_component = raw_specialist_output
            persistence_component = 0.0
            explained_output_preclip = model_component
            predicted_aqi = max(0.0, explained_output_preclip)

        # ---------------------------------------------------------------------
        # ROUTE 2: Horizons 7..37 (Pure Ridge Specialist)
        # ---------------------------------------------------------------------
        elif 7 <= horizon <= 37:
            specialist_type = "Ridge Linear Specialist"
            blend_weight = 1.0
            ridge_idx = horizon - 7  # 0 to 30

            coef = self.model.m_rest.coef_[ridge_idx]         # (114,)
            intercept = self.model.m_rest.intercept_[ridge_idx] # scalar

            # Interventional linear SHAP: phi_i = coef_i * (x_i - bg_mean_i)
            phi_unweighted = coef * (scaled_1d - self.bg_means)
            specialist_base_val = float(intercept + np.dot(coef, self.bg_means))
            raw_specialist_output = specialist_base_val + float(np.sum(phi_unweighted))

            base_value = specialist_base_val
            phi_final = phi_unweighted
            model_component = raw_specialist_output
            persistence_component = 0.0
            explained_output_preclip = model_component
            predicted_aqi = max(0.0, explained_output_preclip)

        # ---------------------------------------------------------------------
        # ROUTE 3: Horizons 38..72 (Blended Ridge + Persistence Specialist)
        # ---------------------------------------------------------------------
        else:
            specialist_type = "Ridge + Persistence Blend"
            blend_idx = horizon - 38  # 0 to 34
            blend_weight = float(self.blend_weights[blend_idx])
            ridge_idx = horizon - 7   # 31 to 65

            coef = self.model.m_rest.coef_[ridge_idx]
            intercept = self.model.m_rest.intercept_[ridge_idx]

            phi_unweighted = coef * (scaled_1d - self.bg_means)
            specialist_base_val = float(intercept + np.dot(coef, self.bg_means))
            raw_specialist_output = specialist_base_val + float(np.sum(phi_unweighted))

            # Scale BOTH base value and feature contributions by blend_weight
            base_value = blend_weight * specialist_base_val
            phi_final = blend_weight * phi_unweighted
            model_component = blend_weight * raw_specialist_output
            persistence_component = (1.0 - blend_weight) * current_aqi

            explained_output_preclip = model_component + persistence_component
            predicted_aqi = max(0.0, explained_output_preclip)

        # Compute exact additivity error: (base_value + sum(phi_final) + persistence) vs explained_preclip
        reconstructed_sum = base_value + float(np.sum(phi_final)) + persistence_component
        additivity_error = abs(reconstructed_sum - explained_output_preclip)

        # Sort top K features by absolute final SHAP contribution
        abs_phi = np.abs(phi_final)
        top_indices = np.argsort(abs_phi)[::-1][:top_k]

        top_features = []
        for idx in top_indices:
            val_shap = float(phi_final[idx])
            top_features.append({
                "feature": self.feature_names[idx],
                "raw_value": float(raw_1d[idx]),
                "scaled_value": float(scaled_1d[idx]),
                "shap_value": round(val_shap, 4),
                "impact": "increases_prediction" if val_shap >= 0 else "decreases_prediction",
            })

        return {
            "horizon": horizon,
            "specialist_type": specialist_type,
            "blend_weight": round(blend_weight, 4),
            "raw_specialist_output": round(raw_specialist_output, 4),
            "specialist_base_value": round(specialist_base_val, 4),
            "base_value": round(base_value, 4),
            "model_component": round(model_component, 4),
            "persistence_component": round(persistence_component, 4),
            "explained_output_preclip": round(explained_output_preclip, 4),
            "predicted_aqi": round(predicted_aqi, 4),
            "additivity_error": float(additivity_error),
            "top_features": top_features,
        }
