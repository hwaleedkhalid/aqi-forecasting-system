"""Pearls AQI Predictor - Feature-Store-Driven Daily Candidate Training & Evaluation Runner.

Automates daily candidate model training from Hopsworks offline Feature Store,
enforcing:
1. Exact physical timestamp semantics for targets (T + h*3600, h=1..72).
2. Strict preservation of the protected production holdout (< 2025-06-07T00:00:00Z).
3. Anti-leakage chronological train/validation splitting with >72h embargo gap.
4. Train-only StandardScaler fitting.
5. Multi-horizon and persistence baseline evaluation.
6. Dynamic recommendation gate (default: retain_champion; never automatic promotion).
7. Isolated candidate directory generation with SHA256 manifest.
8. Complete production champion immutability (data/runtime/production/* is untouched).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_DIR,
    FORECAST_HORIZONS,
    HOPSWORKS_API_KEY,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT,
    MODELS_DIR,
    RUNTIME_DIR,
)
from src.exceptions import FeatureStoreError, ModelTrainingError, ValidationError
from src.feature_pipeline.hopsworks_integration import (
    HopsworksFeatureStoreConnector,
    load_canonical_feature_names,
)
from src.logger import logger
from src.models.hybrid_specialist_model import PersistenceAwareHybridModel
from src.models.lightgbm_models import LightGBMDirectMultiOutput
from src.models.naive_baseline import NaivePersistenceBaseline
from src.models.random_forest_model import RandomForestAQIModel
from src.models.ridge_model import RidgeAQIModel
from src.training_pipeline.dataset_builder import FeatureStoreTrainingLoader
from src.training_pipeline.evaluator import ModelEvaluator

DEFAULT_CANDIDATE_DIR = MODELS_DIR / "candidates"
PRODUCTION_DIR = RUNTIME_DIR / "production"

# EXP-019 Frozen Reference Benchmarks
EXP019_REFERENCE_BENCHMARKS = {
    "model_id": "EXP-019",
    "architecture": "PersistenceAwareHybridModel",
    "canonical_features": 114,
    "holdout_test": {
        "evaluation_protocol": "Chronological Out-of-Time Held-Out Test (2025-06-07 to 2026-08-28)",
        "sample_count": 9311,
        "overall_rmse": 75.91,
        "overall_mae": 53.55,
        "overall_r2": 0.4858,
        "h1_rmse": 50.43,
        "h6_rmse": 62.15,
        "h24_rmse": 72.88,
        "h48_rmse": 75.40,
        "h72_rmse": 77.43,
    },
    "walk_forward": {
        "evaluation_protocol": "4-Fold Seasonal Walk-Forward with 72h Embargo",
        "mean_rmse": 83.44,
        "wins_vs_persistence": "4/4",
        "relative_rmse_gain": "24.46%",
    },
}


def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hex digest of a file (normalizing JSON to LF)."""
    if file_path.suffix.lower() == ".json":
        text = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_extreme_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 200.0) -> dict[str, Any]:
    """Compute error metrics on extreme AQI subsets with sample count and low-n flag."""
    mask = y_true > threshold
    n = int(mask.sum())
    if n == 0:
        return {"threshold": threshold, "sample_count": 0, "rmse": None, "mae": None, "low_sample_warning": True}
    errors = y_pred[mask] - y_true[mask]
    return {
        "threshold": threshold,
        "sample_count": n,
        "rmse": float(round(np.sqrt(np.mean(errors ** 2)), 2)),
        "mae": float(round(np.mean(np.abs(errors)), 2)),
        "low_sample_warning": bool(n < 30),
    }


class DailyCandidateTrainingRunner:
    """Orchestrates Feature-Store-driven candidate training, evaluation, and artifact isolation."""

    def __init__(
        self,
        candidate_family: str = "ridge",
        output_dir: Path | None = None,
        location_id: str = "lahore",
        forecast_horizons: int = FORECAST_HORIZONS,
        read_timeout: int = 300,
        train_ratio: float = 0.8,
    ) -> None:
        """Initialize candidate training runner.

        Args:
            candidate_family: Model family to train ('ridge', 'hybrid', 'lightgbm', 'random_forest').
            output_dir: Base directory for candidate run outputs.
            location_id: Feature Store location entity.
            forecast_horizons: Number of prediction horizons (72).
            read_timeout: Bounded timeout for Hopsworks offline read (seconds).
            train_ratio: Chronological train split fraction (0.8).
        """
        self.candidate_family = candidate_family.lower()
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_CANDIDATE_DIR
        self.location_id = location_id
        self.forecast_horizons = forecast_horizons
        self.read_timeout = read_timeout
        self.train_ratio = train_ratio

        # Production Champion Safety Invariant
        self._assert_production_protection()

        self.loader = FeatureStoreTrainingLoader(
            location_id=location_id,
            forecast_horizons=forecast_horizons,
            read_timeout=read_timeout,
        )

    def _assert_production_protection(self) -> None:
        """Strictly assert that candidate outputs never target production directories."""
        prod_resolved = PRODUCTION_DIR.resolve()
        out_resolved = self.output_dir.resolve()
        if out_resolved == prod_resolved or prod_resolved in out_resolved.parents:
            raise ValidationError(
                f"Production safety violation: candidate output directory '{out_resolved}' "
                f"cannot be within production directory '{prod_resolved}'."
            )

    def _build_candidate_model(self) -> Any:
        """Instantiate candidate model family."""
        if self.candidate_family == "ridge":
            # Candidate Ridge model uses L2 regularization alpha=10.0 tuned for 113 collinear features
            return MultiOutputRegressor(Ridge(alpha=10.0, random_state=42), n_jobs=-1)
        elif self.candidate_family == "hybrid":
            return PersistenceAwareHybridModel(
                short_estimators=40,
                short_lr=0.1,
                ridge_alpha=1.0,
                min_blend_weight=0.7,
            )
        elif self.candidate_family == "lightgbm":
            return LightGBMDirectMultiOutput(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
            )
        elif self.candidate_family == "random_forest":
            return RandomForestAQIModel(
                n_estimators=50,
                max_depth=12,
                random_state=42,
            )
        else:
            raise ValidationError(
                f"Unknown candidate model family '{self.candidate_family}'. "
                f"Supported: 'ridge', 'hybrid', 'lightgbm', 'random_forest'."
            )

    def run(
        self,
        dry_run: bool = False,
        offline_data_path: Path | None = None,
    ) -> dict[str, Any]:
        """Execute full candidate training and evaluation pipeline.

        Args:
            dry_run: If True, validates data, targets, and splits without model training.
            offline_data_path: Optional local CSV path for offline testing/CI.

        Returns:
            Candidate execution report dictionary.
        """
        start_time = time.time()
        run_timestamp = datetime.now(timezone.utc)
        run_id = f"candidate-{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}-{self.candidate_family}"

        candidate_run_dir = self.output_dir / run_id
        candidate_run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== Starting Candidate Training Run: {run_id} ===")
        logger.info(f"Model family: {self.candidate_family}, Dry run: {dry_run}")

        # 1. Fetch historical data from Feature Store or local offline file
        query_meta = {}
        if offline_data_path is not None and Path(offline_data_path).exists():
            logger.info(f"Loading offline dataset from local file: {offline_data_path}")
            raw_df = pd.read_csv(offline_data_path)
            query_meta = {
                "source": "Local Offline File (Testing)",
                "path": str(offline_data_path),
                "raw_row_count": len(raw_df),
                "latest_materialized_dt": int(raw_df["dt"].max()) if "dt" in raw_df.columns else 0,
                "materialized_age_hours": 0.0,
            }
        else:
            raw_df, query_meta = self.loader.fetch_offline_data()

        # 2. Schema audit, finiteness checks, and auditable deduplication
        cleaned_df, audit_report = self.loader.audit_and_clean_data(raw_df)

        # 3. Restrict candidate development to pre-holdout observations (< 2025-06-07T00:00:00Z)
        dev_df, holdout_meta = self.loader.filter_candidate_development_data(cleaned_df)

        # 4. Construct exact physical timestamp targets (T + h*3600)
        df_features, df_targets = self.loader.construct_physical_targets(dev_df)

        # 5. Chronological 80/20 train/val split with >72h embargo gap
        split_data = self.loader.split_chronological_with_embargo(
            df_features, df_targets, train_ratio=self.train_ratio
        )
        leakage_audit = split_data["leakage_audit"]

        # Feature columns: all 114 canonical features excluding 'dt'
        canonical_114 = self.loader.canonical_features
        feature_cols = [c for c in canonical_114 if c != "dt"]

        X_train_raw = split_data["X_train_df"][feature_cols].values.astype(np.float32)
        y_train = split_data["Y_train_df"].values.astype(np.float32)

        X_val_raw = split_data["X_val_df"][feature_cols].values.astype(np.float32)
        y_val = split_data["Y_val_df"].values.astype(np.float32)

        current_aqi_val = split_data["X_val_df"]["epa_aqi"].values.astype(np.float64)

        # Assemble dataset provenance
        provenance = {
            "run_id": run_id,
            "generated_at": run_timestamp.isoformat(),
            "source": query_meta.get("source", "Hopsworks Feature Store"),
            "feature_group": getattr(self.loader.connector, "feature_group_name", "aqi_weather_features_v2"),
            "feature_group_version": getattr(self.loader.connector, "feature_group_version", 1),
            "location_id": self.location_id,
            "total_feature_store_records": query_meta.get("raw_row_count", len(raw_df)),
            "latest_offline_materialized_dt": query_meta.get("latest_materialized_dt"),
            "latest_materialized_utc": query_meta.get("latest_materialized_utc"),
            "materialization_age_hours": query_meta.get("materialized_age_hours"),
            "cleaned_records": audit_report["cleaned_row_count"],
            "identical_duplicates_dropped": audit_report["identical_duplicates_dropped"],
            "temporal_range_total": {
                "earliest_dt": audit_report["earliest_dt"],
                "latest_dt": audit_report["latest_dt"],
                "earliest_utc": audit_report["earliest_utc"],
                "latest_utc": audit_report["latest_utc"],
            },
            "protected_holdout": holdout_meta,
            "post_cutoff_quarantined_region": holdout_meta,
            "usable_development_samples": len(df_features),
            "train_sample_count": len(X_train_raw),
            "val_sample_count": len(X_val_raw),
            "embargo_gap_hours": self.forecast_horizons,
            "canonical_features_count": len(canonical_114),
            "fitted_predictor_count": len(feature_cols),
            "leakage_audit": leakage_audit,
        }

        provenance_file = candidate_run_dir / "dataset_provenance.json"
        with open(provenance_file, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2)

        if dry_run:
            dry_report = {
                "run_id": run_id,
                "status": "dry_run_success",
                "mode": "dry_run",
                "candidate_family": self.candidate_family,
                "dataset_provenance": provenance,
                "production_champion_status": "UNTOUCHED_READ_ONLY",
                "runtime_seconds": round(time.time() - start_time, 2),
            }
            report_file = candidate_run_dir / "candidate_evaluation_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(dry_report, f, indent=2)
            logger.info(f"Dry run complete. Report saved to {report_file}")
            return dry_report

        # 6. Fit StandardScaler EXCLUSIVELY on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        scaler_file = candidate_run_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_file)
        logger.info(f"Fitted train-only StandardScaler saved to {scaler_file}")

        # 7. Train Candidate Model
        logger.info(f"Fitting candidate model family '{self.candidate_family}' on {len(X_train_scaled)} samples...")
        t_fit_start = time.time()
        model = self._build_candidate_model()
        model.fit(X_train_scaled, y_train)
        training_duration = round(time.time() - t_fit_start, 2)
        logger.info(f"Model fit complete in {training_duration}s.")

        model_file = candidate_run_dir / "candidate_model.joblib"
        joblib.dump(model, model_file)
        logger.info(f"Candidate model artifact saved to {model_file}")

        # 8. Generate Predictions on Validation Set
        if isinstance(model, PersistenceAwareHybridModel):
            y_pred = model.predict(X_val_scaled, current_aqi=current_aqi_val)
        else:
            y_pred = model.predict(X_val_scaled)

        # 9. Evaluate Candidate Metrics
        candidate_eval = ModelEvaluator.evaluate_predictions(
            y_val, y_pred, model_name=f"Candidate-{self.candidate_family.upper()}"
        )

        # Extreme Event Metrics
        candidate_eval["extreme_events"] = {
            "severe_gt200": compute_extreme_metrics(y_val, y_pred, threshold=200.0),
            "hazardous_gt300": compute_extreme_metrics(y_val, y_pred, threshold=300.0),
        }

        # 10. Evaluate Naive Persistence Baseline on the exact same validation split
        baseline = NaivePersistenceBaseline(forecast_horizons=self.forecast_horizons)
        y_pred_baseline = baseline.predict_from_current(current_aqi_val)
        baseline_eval = ModelEvaluator.evaluate_predictions(
            y_val, y_pred_baseline, model_name="Naive Persistence Baseline"
        )
        baseline_eval["extreme_events"] = {
            "severe_gt200": compute_extreme_metrics(y_val, y_pred_baseline, threshold=200.0),
            "hazardous_gt300": compute_extreme_metrics(y_val, y_pred_baseline, threshold=300.0),
        }

        # % Gain vs Baseline
        baseline_rmse = baseline_eval["overall_rmse"]
        cand_rmse = candidate_eval["overall_rmse"]
        pct_improvement_vs_baseline = round(((baseline_rmse - cand_rmse) / baseline_rmse) * 100.0, 2)

        evaluation_file = candidate_run_dir / "evaluation.json"
        with open(evaluation_file, "w", encoding="utf-8") as f:
            json.dump({
                "candidate_evaluation": candidate_eval,
                "baseline_evaluation": baseline_eval,
                "improvement_vs_baseline_pct": pct_improvement_vs_baseline,
            }, f, indent=2)

        # 11. Dynamic Recommendation Gate (Strict Same-Protocol Multi-Condition Logic)
        # Default: retain champion.
        # Candidate merits manual review ONLY on same-protocol evidence:
        # 1. Candidate beats persistence overall (cand_rmse < baseline_rmse)
        # 2. Candidate beats persistence at milestone horizons h1, h24, and h72
        # 3. Candidate achieves a meaningful relative improvement over persistence (>= 15%)
        # 4. No material regression in AQI >200 / >300 subsets when sample counts are sufficient (>= 30)
        # Under NO circumstances does the gate use EXP-019 out-of-time holdout RMSE (75.91)
        # because the candidate is evaluated on the pre-holdout development validation split.
        beats_persistence_overall = bool(cand_rmse < baseline_rmse)
        beats_persistence_h1 = bool(
            candidate_eval["key_horizons"]["h1"]["rmse"] < baseline_eval["key_horizons"]["h1"]["rmse"]
        )
        beats_persistence_h24 = bool(
            candidate_eval["key_horizons"]["h24"]["rmse"] < baseline_eval["key_horizons"]["h24"]["rmse"]
        )
        beats_persistence_h72 = bool(
            candidate_eval["key_horizons"]["h72"]["rmse"] < baseline_eval["key_horizons"]["h72"]["rmse"]
        )
        meaningful_improvement = bool(pct_improvement_vs_baseline >= 15.0)

        # Check for material regression on extreme subsets (>200 and >300) when sample count is sufficient (>= 30)
        no_extreme_regression = True
        extreme_regression_notes = []
        for thresh_key, thresh_name in [("severe_gt200", "AQI > 200"), ("hazardous_gt300", "AQI > 300")]:
            c_ext = candidate_eval["extreme_events"].get(thresh_key, {})
            b_ext = baseline_eval["extreme_events"].get(thresh_key, {})
            n_samples = c_ext.get("sample_count", 0)
            if n_samples >= 30 and c_ext.get("rmse") is not None and b_ext.get("rmse") is not None:
                if c_ext["rmse"] > b_ext["rmse"] * 1.05:
                    no_extreme_regression = False
                    extreme_regression_notes.append(
                        f"{thresh_name} regression: candidate RMSE ({c_ext['rmse']}) > baseline RMSE ({b_ext['rmse']})"
                    )

        merits_manual_review = (
            beats_persistence_overall
            and beats_persistence_h1
            and beats_persistence_h24
            and beats_persistence_h72
            and meaningful_improvement
            and no_extreme_regression
        )

        if merits_manual_review:
            recommendation = "manual_review_recommended"
            recommendation_reason = (
                f"Candidate ({self.candidate_family}) achieved {pct_improvement_vs_baseline}% gain vs persistence, "
                "beat persistence across h1, h24, h72 without extreme subset regressions on development validation. "
                "Candidate merits manual human review (NO automatic promotion)."
            )
        else:
            recommendation = "retain_champion"
            reasons = []
            if not beats_persistence_overall:
                reasons.append("did not beat persistence overall")
            if not (beats_persistence_h1 and beats_persistence_h24 and beats_persistence_h72):
                reasons.append("failed to beat persistence across all key horizons (h1, h24, h72)")
            if not meaningful_improvement:
                reasons.append(f"improvement vs persistence ({pct_improvement_vs_baseline}%) below 15% threshold")
            if not no_extreme_regression:
                reasons.extend(extreme_regression_notes)
            recommendation_reason = (
                f"Candidate did not satisfy same-protocol review criteria: {'; '.join(reasons)}. "
                "EXP-019 retained as production champion."
            )

        comparison_report = {
            "run_id": run_id,
            "candidate_family": self.candidate_family,
            "validation_protocol": f"Pre-Holdout Chronological Split ({self.train_ratio*100:.0f}/{(1-self.train_ratio)*100:.0f}) with 72h Embargo",
            "candidate_metrics": {
                "overall_rmse": candidate_eval["overall_rmse"],
                "overall_mae": candidate_eval["overall_mae"],
                "overall_r2": candidate_eval["overall_r2"],
                "h1_rmse": candidate_eval["key_horizons"]["h1"]["rmse"],
                "h6_rmse": candidate_eval["key_horizons"]["h6"]["rmse"],
                "h24_rmse": candidate_eval["key_horizons"]["h24"]["rmse"],
                "h48_rmse": candidate_eval["key_horizons"]["h48"]["rmse"],
                "h72_rmse": candidate_eval["key_horizons"]["h72"]["rmse"],
            },
            "baseline_metrics": {
                "overall_rmse": baseline_eval["overall_rmse"],
                "overall_mae": baseline_eval["overall_mae"],
                "overall_r2": baseline_eval["overall_r2"],
                "h1_rmse": baseline_eval["key_horizons"]["h1"]["rmse"],
                "h24_rmse": baseline_eval["key_horizons"]["h24"]["rmse"],
                "h72_rmse": baseline_eval["key_horizons"]["h72"]["rmse"],
            },
            "improvement_vs_baseline_pct": pct_improvement_vs_baseline,
            "same_protocol_evaluation": {
                "beats_persistence_overall": beats_persistence_overall,
                "beats_persistence_h1": beats_persistence_h1,
                "beats_persistence_h24": beats_persistence_h24,
                "beats_persistence_h72": beats_persistence_h72,
                "meaningful_improvement_ge_15pct": meaningful_improvement,
                "no_extreme_regression": no_extreme_regression,
            },
            "exp019_reference_benchmarks": EXP019_REFERENCE_BENCHMARKS,
            "comparable_evaluation_protocol": False,
            "protocol_difference_note": (
                "Candidate evaluated on pre-2025-06-07 development validation split; "
                "EXP-019 holdout metrics are from the protected 2025-06-07 to 2026-08-28 holdout partition."
            ),
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
            "automatic_promotion_attempted": False,
            "production_champion_unmodified": True,
        }

        comparison_file = candidate_run_dir / "candidate_comparison.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2)

        # 12. Build Artifact Checksums Manifest
        manifest = {
            "run_id": run_id,
            "created_at": run_timestamp.isoformat(),
            "candidate_family": self.candidate_family,
            "artifacts": {
                "candidate_model.joblib": compute_sha256(model_file),
                "scaler.joblib": compute_sha256(scaler_file),
                "evaluation.json": compute_sha256(evaluation_file),
                "dataset_provenance.json": compute_sha256(provenance_file),
                "candidate_comparison.json": compute_sha256(comparison_file),
            },
        }
        manifest_file = candidate_run_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        total_runtime = round(time.time() - start_time, 2)
        summary_report = {
            "run_id": run_id,
            "status": "candidate_training_completed",
            "candidate_family": self.candidate_family,
            "recommendation": recommendation,
            "recommendation_reason": recommendation_reason,
            "candidate_rmse": candidate_eval["overall_rmse"],
            "candidate_mae": candidate_eval["overall_mae"],
            "candidate_r2": candidate_eval["overall_r2"],
            "baseline_rmse": baseline_eval["overall_rmse"],
            "pct_gain_vs_baseline": pct_improvement_vs_baseline,
            "training_duration_seconds": training_duration,
            "total_runtime_seconds": total_runtime,
            "candidate_run_dir": str(candidate_run_dir),
            "production_champion": {
                "model_id": "EXP-019",
                "status": "FROZEN_READ_ONLY",
                "unmodified": True,
            },
        }

        # Backward-compatible report at output_dir root
        compat_file = candidate_run_dir / "candidate_evaluation_report.json"
        with open(compat_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)

        logger.info(
            f"=== Candidate Training Run Complete ({run_id}) ===\n"
            f"Candidate RMSE={candidate_eval['overall_rmse']:.2f}, Baseline RMSE={baseline_eval['overall_rmse']:.2f} "
            f"({pct_improvement_vs_baseline:+.2f}% gain). Recommendation: {recommendation}."
        )
        return summary_report


def main() -> int:
    """CLI entry point for daily candidate training."""
    parser = argparse.ArgumentParser(description="Feature-Store-driven daily candidate training & evaluation.")
    parser.add_argument(
        "--candidate-family",
        type=str,
        default="ridge",
        choices=["ridge", "hybrid", "lightgbm", "random_forest"],
        help="Model family to train (default: ridge).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/candidates",
        help="Base directory for candidate runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify Hopsworks retrieval, target generation, and splitting without training models.",
    )
    parser.add_argument(
        "--offline-data",
        type=str,
        default=None,
        help="Path to local CSV for offline testing/CI without connecting to Hopsworks.",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=300,
        help="Bounded timeout in seconds for Hopsworks Arrow Flight offline read (default: 300).",
    )
    args = parser.parse_args()

    try:
        runner = DailyCandidateTrainingRunner(
            candidate_family=args.candidate_family,
            output_dir=Path(args.output_dir),
            read_timeout=args.read_timeout,
        )
        offline_p = Path(args.offline_data) if args.offline_data else None
        res = runner.run(dry_run=args.dry_run, offline_data_path=offline_p)
        print(json.dumps(res, indent=2))
        return 0
    except Exception as e:
        logger.error(f"Daily candidate training failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
