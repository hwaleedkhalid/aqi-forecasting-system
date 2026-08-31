"""Pearls AQI Predictor - Multi-Horizon Model Evaluator.

Computes overall and per-horizon regression metrics (RMSE, MAE, R²) and benchmark
improvements against the Naive Persistence Baseline.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODELS_DIR
from src.logger import logger


class ModelEvaluator:
    """Evaluates multi-output forecasting models across all 72 prediction horizons."""

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
    ) -> dict[str, Any]:
        """Compute overall and horizon-specific regression evaluation metrics.

        Args:
            y_true: Ground truth target matrix of shape (n_samples, n_horizons).
            y_pred: Predicted target matrix of shape (n_samples, n_horizons).
            model_name: Model identifier name.

        Returns:
            Dictionary containing overall and per-horizon metrics.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n_samples, n_horizons = y_true.shape

        # Overall aggregate metrics across all horizons and samples
        overall_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        overall_mae = float(mean_absolute_error(y_true, y_pred))
        overall_r2 = float(r2_score(y_true, y_pred))

        # Per-horizon metrics for every hourly step h=1..72
        horizon_metrics = {}
        for h in range(n_horizons):
            h_step = h + 1
            h_true = y_true[:, h]
            h_pred = y_pred[:, h]
            horizon_metrics[f"h{h_step}"] = {
                "step": h_step,
                "rmse": float(np.sqrt(mean_squared_error(h_true, h_pred))),
                "mae": float(mean_absolute_error(h_true, h_pred)),
                "r2": float(r2_score(h_true, h_pred)),
            }

        # Key benchmark milestone horizons
        key_steps = [1, 3, 6, 12, 24, 48, 72]
        key_horizons = {
            f"h{k}": horizon_metrics[f"h{k}"]
            for k in key_steps
            if f"h{k}" in horizon_metrics
        }

        logger.info(
            f"[{model_name}] Evaluation Results: "
            f"RMSE={overall_rmse:.2f}, MAE={overall_mae:.2f}, R²={overall_r2:.4f} "
            f"(h+1 RMSE: {horizon_metrics.get('h1', {}).get('rmse', 0):.2f}, "
            f"h+24 RMSE: {horizon_metrics.get('h24', {}).get('rmse', 0):.2f}, "
            f"h+72 RMSE: {horizon_metrics.get('h72', {}).get('rmse', 0):.2f})"
        )

        return {
            "model_name": model_name,
            "overall_rmse": overall_rmse,
            "overall_mae": overall_mae,
            "overall_r2": overall_r2,
            "key_horizons": key_horizons,
            "all_horizons": horizon_metrics,
        }

    @staticmethod
    def compare_models(
        evaluations: dict[str, dict[str, Any]],
        baseline_name: str = "Naive Persistence Baseline",
    ) -> pd.DataFrame:
        """Construct comparative evaluation summary table with baseline improvements.

        Args:
            evaluations: Dictionary mapping model names to evaluation output dictionaries.
            baseline_name: Identifier of benchmark baseline model.

        Returns:
            Formatted DataFrame comparing all evaluated models.
        """
        records = []
        baseline_rmse = None

        if baseline_name in evaluations:
            baseline_rmse = evaluations[baseline_name]["overall_rmse"]

        for m_name, ev in evaluations.items():
            rmse = ev["overall_rmse"]
            mae = ev["overall_mae"]
            r2 = ev["overall_r2"]

            # Calculate % RMSE improvement relative to baseline
            pct_improvement = 0.0
            if baseline_rmse and baseline_rmse > 0 and m_name != baseline_name:
                pct_improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100.0

            records.append({
                "Model": m_name,
                "Overall RMSE": round(rmse, 2),
                "Overall MAE": round(mae, 2),
                "Overall R²": round(r2, 4),
                "h+1 RMSE": round(ev["all_horizons"].get("h1", {}).get("rmse", 0), 2),
                "h+24 RMSE": round(ev["all_horizons"].get("h24", {}).get("rmse", 0), 2),
                "h+72 RMSE": round(ev["all_horizons"].get("h72", {}).get("rmse", 0), 2),
                "vs Baseline (% RMSE)": f"{pct_improvement:+.2f}%" if m_name != baseline_name else "Baseline",
            })

        df_comp = pd.DataFrame(records).sort_values("Overall RMSE").reset_index(drop=True)
        return df_comp

    @staticmethod
    def save_comparison(
        evaluations: dict[str, dict[str, Any]],
        output_path: Path = MODELS_DIR / "model_comparison.json",
    ) -> Path:
        """Serialize model comparison metrics to JSON.

        Args:
            evaluations: Evaluation dictionary.
            output_path: Destination JSON path.

        Returns:
            Path to saved comparison report.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evaluations, f, indent=2)
        logger.info(f"Saved model comparison report to {output_path}")
        return output_path
