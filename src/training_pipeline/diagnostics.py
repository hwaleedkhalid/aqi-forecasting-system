"""Pearls AQI Predictor - Diagnostic Analysis Engine.

Performs deep diagnostics on dataset distribution shifts, seasonal performance,
AQI category segmentation, multi-horizon error decay, and catastrophic failure modes
without modifying models or touching production evaluation loops.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.logger import logger


class DatasetDiagnostics:
    """Computes distribution statistics, temporal shifts, and segmented error metrics."""

    def __init__(
        self,
        data_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        """Initialize DatasetDiagnostics.

        Args:
            data_dir: Path to directory containing processed datasets and timestamp metadata.
            models_dir: Path to directory containing model artifacts and predictions.
        """
        self.data_dir = data_dir
        self.models_dir = models_dir

    def load_historical_data(self) -> pd.DataFrame:
        """Load clean historical AQI dataframe with parsed UTC timestamps."""
        csv_path = self.data_dir / "historical_aqi_clean.csv"
        df = pd.read_csv(csv_path)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
        return df

    def compute_distribution_shift(self) -> dict[str, Any]:
        """Compute comprehensive summary statistics and distribution shift between Train and Test sets.

        Returns:
            Dictionary containing descriptive statistics, extreme value percentiles,
            and EPA category distributions for train and test partitions.
        """
        summary_path = self.data_dir / "dataset_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        split_ts = pd.to_datetime(summary["split_timestamp_utc"])
        embargo_start = pd.to_datetime(summary["train_period"]["end_utc"])

        df = self.load_historical_data()
        train_df = df[df["datetime_utc"] <= embargo_start].copy()
        test_df = df[df["datetime_utc"] >= split_ts].copy()

        def _stats(series: pd.Series) -> dict[str, float]:
            return {
                "count": int(len(series)),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "p99": float(series.quantile(0.99)),
                "max": float(series.max()),
                "min": float(series.min()),
            }

        def _cat_proportions(series: pd.Series) -> dict[str, float]:
            counts = series.value_counts(normalize=True).to_dict()
            return {k: round(float(v) * 100.0, 2) for k, v in counts.items()}

        report = {
            "sample_counts": {
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "train_start": str(train_df["datetime_utc"].min()),
                "train_end": str(train_df["datetime_utc"].max()),
                "test_start": str(test_df["datetime_utc"].min()),
                "test_end": str(test_df["datetime_utc"].max()),
            },
            "epa_aqi_distribution": {
                "train": _stats(train_df["epa_aqi"]),
                "test": _stats(test_df["epa_aqi"]),
            },
            "pm2_5_distribution": {
                "train": _stats(train_df["pm2_5"]),
                "test": _stats(test_df["pm2_5"]),
            },
            "aqi_category_proportions_pct": {
                "train": _cat_proportions(train_df["aqi_category"]),
                "test": _cat_proportions(test_df["aqi_category"]),
            },
            "dominant_pollutant_proportions_pct": {
                "train": _cat_proportions(train_df["dominant_pollutant"]),
                "test": _cat_proportions(test_df["dominant_pollutant"]),
            },
        }
        return report

    def evaluate_segmented_performance(self) -> dict[str, Any]:
        """Evaluate segmented performance (by Season and AQI Severity) for Naive Baseline and Ridge Regression.

        Returns:
            Dictionary containing segmented RMSE, MAE, and sample counts across subsets.
        """
        # Load test arrays and timestamps
        y_test = np.load(self.data_dir / "y_test.npy")
        current_test = np.load(self.data_dir / "current_aqi_test.npy")
        X_test = np.load(self.data_dir / "X_test.npy")
        test_ts_df = pd.read_csv(self.data_dir / "test_timestamps.csv")
        test_ts = pd.to_datetime(test_ts_df["datetime_utc"])

        # Reconstruct predictions
        # 1. Naive Baseline: repeat current AQI across all 72 horizons
        y_pred_naive = np.tile(current_test[:, np.newaxis], (1, 72))

        # 2. Ridge Regression
        import joblib
        ridge_path = self.models_dir / "ridge_model.joblib"
        ridge_model = joblib.load(ridge_path)
        y_pred_ridge = ridge_model.predict(X_test)

        # 3. Categorize Seasons
        months = test_ts.dt.month.values
        # Winter Smog: Nov (11), Dec (12), Jan (1), Feb (2)
        # Pre-Winter / Autumn: Sep (9), Oct (10)
        # Spring / Summer: Mar (3), Apr (4), May (5), Jun (6)
        # Monsoon: Jul (7), Aug (8)
        season_masks = {
            "Winter_Smog (Nov-Feb)": np.isin(months, [11, 12, 1, 2]),
            "Pre_Winter (Sep-Oct)": np.isin(months, [9, 10]),
            "Spring_Summer (Mar-Jun)": np.isin(months, [3, 4, 5, 6]),
            "Monsoon (Jul-Aug)": np.isin(months, [7, 8]),
        }

        # 4. Categorize by AQI Severity at prediction time t (Current AQI)
        current_masks = {
            "Current_Good_Moderate (<=100)": current_test <= 100,
            "Current_USG_Unhealthy (101-200)": (current_test > 100) & (current_test <= 200),
            "Current_Very_Unhealthy (201-300)": (current_test > 200) & (current_test <= 300),
            "Current_Hazardous (>300)": current_test > 300,
        }

        def _calc_metrics(true_sub: np.ndarray, pred_sub: np.ndarray) -> dict[str, float]:
            if len(true_sub) == 0:
                return {"rmse": 0.0, "mae": 0.0, "count": 0}
            return {
                "count": int(len(true_sub)),
                "overall_rmse": round(float(np.sqrt(mean_squared_error(true_sub, pred_sub))), 2),
                "overall_mae": round(float(mean_absolute_error(true_sub, pred_sub)), 2),
                "h1_rmse": round(float(np.sqrt(mean_squared_error(true_sub[:, 0], pred_sub[:, 0]))), 2),
                "h24_rmse": round(float(np.sqrt(mean_squared_error(true_sub[:, 23], pred_sub[:, 23]))), 2),
                "h72_rmse": round(float(np.sqrt(mean_squared_error(true_sub[:, 71], pred_sub[:, 71]))), 2),
            }

        seasonal_metrics = {}
        for s_name, mask in season_masks.items():
            seasonal_metrics[s_name] = {
                "Naive_Baseline": _calc_metrics(y_test[mask], y_pred_naive[mask]),
                "Ridge_Regression": _calc_metrics(y_test[mask], y_pred_ridge[mask]),
            }

        current_severity_metrics = {}
        for c_name, mask in current_masks.items():
            current_severity_metrics[c_name] = {
                "Naive_Baseline": _calc_metrics(y_test[mask], y_pred_naive[mask]),
                "Ridge_Regression": _calc_metrics(y_test[mask], y_pred_ridge[mask]),
            }

        # 5. Find Catastrophic Error Outliers (Top 1% largest absolute errors in Ridge)
        abs_errors = np.abs(y_test - y_pred_ridge)
        max_errors_per_sample = np.max(abs_errors, axis=1)  # max error across 72h
        threshold_99 = np.percentile(max_errors_per_sample, 99)
        catastrophic_idx = np.where(max_errors_per_sample >= threshold_99)[0]

        catastrophic_events = []
        for idx in catastrophic_idx[:15]:  # top 15 events
            catastrophic_events.append({
                "timestamp_utc": str(test_ts.iloc[idx]),
                "current_aqi": float(current_test[idx]),
                "target_aqi_h24": float(y_test[idx, 23]),
                "target_aqi_h72": float(y_test[idx, 71]),
                "ridge_pred_h24": float(y_pred_ridge[idx, 23]),
                "ridge_pred_h72": float(y_pred_ridge[idx, 71]),
                "max_absolute_error": round(float(max_errors_per_sample[idx]), 2),
            })

        return {
            "seasonal_performance": seasonal_metrics,
            "current_aqi_severity_performance": current_severity_metrics,
            "catastrophic_error_threshold_p99": round(float(threshold_99), 2),
            "top_catastrophic_events": catastrophic_events,
        }

    def generate_diagnostic_report(self, output_path: Path | None = None) -> dict[str, Any]:
        """Generate comprehensive diagnostic report combining distribution shift and segmented errors.

        Args:
            output_path: Optional file path to save JSON report.

        Returns:
            Dictionary containing full diagnostic report.
        """
        logger.info("Executing diagnostic analysis: distribution shift...")
        dist_shift = self.compute_distribution_shift()

        logger.info("Executing diagnostic analysis: segmented performance...")
        seg_perf = self.evaluate_segmented_performance()

        report = {
            "phase": "10.5A - Diagnostic Analysis",
            "distribution_shift": dist_shift,
            "segmented_performance": seg_perf,
        }

        if output_path is None:
            output_path = self.data_dir / "diagnostic_report.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Diagnostic report saved successfully to {output_path}")
        return report
