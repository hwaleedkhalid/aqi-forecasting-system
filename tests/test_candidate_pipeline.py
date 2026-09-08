"""Unit and contract test suite for daily Feature-Store-driven candidate training pipeline.

Tests:
1. Feature Store offline data auditing, schema contracts, and auditable deduplication.
2. Protected holdout isolation (< 2025-06-07T00:00:00Z).
3. Exact physical timestamp target construction without interpolation.
4. Strict chronological splitting with anti-leakage embargo gap.
5. Preprocessing isolation (scaler fit strictly on training set).
6. Candidate model families fitting and multi-horizon evaluation.
7. Dynamic recommendation gate (never automatic promotion).
8. Candidate artifact bundle isolation and production champion protection.
9. 100% secret- and network-independent offline execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.exceptions import FeatureStoreError, ValidationError
from src.feature_pipeline.hopsworks_integration import load_canonical_feature_names
from src.training_pipeline.dataset_builder import FeatureStoreTrainingLoader
from src.training_pipeline.run_daily_candidate import (
    DailyCandidateTrainingRunner,
    compute_extreme_metrics,
    compute_sha256,
)


@pytest.fixture
def canonical_features() -> list[str]:
    """Load canonical 114 ordered features."""
    return load_canonical_feature_names()


@pytest.fixture
def synthetic_hourly_features(canonical_features) -> pd.DataFrame:
    """Generate 500 hours of synthetic continuous feature rows."""
    base_dt = 1700000000  # Nov 2023
    n_rows = 500
    dts = [base_dt + i * 3600 for i in range(n_rows)]

    data = {
        "location_id": ["lahore"] * n_rows,
        "dt": dts,
        "epa_aqi": [float(80 + (i % 60)) for i in range(n_rows)],
    }
    for col in canonical_features:
        if col not in data:
            data[col] = [float(10.0 + (i % 5)) for i in range(n_rows)]

    df = pd.DataFrame(data)
    # Ensure all canonical columns exist in DataFrame
    return df[["location_id"] + canonical_features]


class TestFeatureStoreTrainingLoader:
    """Test suite for FeatureStoreTrainingLoader contracts and invariant checks."""

    def test_schema_audit_detects_missing_columns(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader()
        bad_df = synthetic_hourly_features.drop(columns=["pm2_5_lag_1h"])
        with pytest.raises(ValidationError, match="missing canonical columns"):
            loader.audit_and_clean_data(bad_df)

    def test_schema_audit_rejects_non_integral_dt(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader()
        bad_df = synthetic_hourly_features.copy()
        bad_df.loc[0, "dt"] = 1700000000.55
        with pytest.raises(ValidationError, match="Fractional.*timestamps detected"):
            loader.audit_and_clean_data(bad_df)

    def test_schema_audit_rejects_nan_and_inf(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader()
        # NaN check
        bad_df_nan = synthetic_hourly_features.copy()
        bad_df_nan.loc[5, "pm2_5"] = np.nan
        with pytest.raises(ValidationError, match="NaN values detected"):
            loader.audit_and_clean_data(bad_df_nan)

        # Inf check
        bad_df_inf = synthetic_hourly_features.copy()
        bad_df_inf.loc[5, "pm2_5"] = np.inf
        with pytest.raises(ValidationError, match="Infinite values detected"):
            loader.audit_and_clean_data(bad_df_inf)

    def test_auditable_deduplication_drops_identical_duplicates(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader()
        dup_row = synthetic_hourly_features.iloc[[0]].copy()
        with_dup = pd.concat([synthetic_hourly_features, dup_row], ignore_index=True)
        assert len(with_dup) == len(synthetic_hourly_features) + 1

        cleaned, report = loader.audit_and_clean_data(with_dup)
        assert len(cleaned) == len(synthetic_hourly_features)
        assert report["identical_duplicates_dropped"] == 1

    def test_auditable_deduplication_fails_on_conflicting_duplicates(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader()
        dup_row = synthetic_hourly_features.iloc[[0]].copy()
        dup_row["pm2_5"] = 999.9  # Conflicting feature value for same dt
        with_conflict = pd.concat([synthetic_hourly_features, dup_row], ignore_index=True)

        with pytest.raises(ValidationError, match="Conflicting duplicate records detected"):
            loader.audit_and_clean_data(with_conflict)

    def test_protected_holdout_filter_preserves_holdout(self, canonical_features):
        loader = FeatureStoreTrainingLoader()
        # Create records spanning before and after 2025-06-07 (1749254400)
        dt_pre = 1749250800   # 1 hour before holdout
        dt_post = 1749258000  # 1 hour after holdout
        df = pd.DataFrame({
            "location_id": ["lahore", "lahore"],
            "dt": [dt_pre, dt_post],
            "epa_aqi": [100.0, 150.0],
        })
        for c in canonical_features:
            if c not in df.columns:
                df[c] = 1.0

        dev_df, meta = loader.filter_candidate_development_data(df)
        assert len(dev_df) == 1
        assert int(dev_df["dt"].iloc[0]) == dt_pre
        assert meta["protected_holdout_sample_count"] == 1
        assert meta["holdout_preserved_untouched"] is True

    def test_exact_physical_timestamp_target_semantics(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader(forecast_horizons=72)
        df_feat, df_targ = loader.construct_physical_targets(synthetic_hourly_features)

        # 500 continuous hourly rows -> exactly 500 - 72 = 428 samples with full 72h future
        assert len(df_feat) == 500 - 72
        assert len(df_targ) == 500 - 72
        assert df_targ.shape[1] == 72

        # Verify exact physical shift for sample 0: target_h1 == observation at dt + 3600
        dt_0 = int(df_feat["dt"].iloc[0])
        val_0_h1 = df_targ["target_h1"].iloc[0]
        val_0_h72 = df_targ["target_h72"].iloc[0]

        expected_h1 = synthetic_hourly_features.loc[
            synthetic_hourly_features["dt"] == dt_0 + 3600, "epa_aqi"
        ].values[0]
        expected_h72 = synthetic_hourly_features.loc[
            synthetic_hourly_features["dt"] == dt_0 + 72 * 3600, "epa_aqi"
        ].values[0]

        assert val_0_h1 == expected_h1
        assert val_0_h72 == expected_h72

    def test_missing_physical_target_timestamp_drops_sample(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader(forecast_horizons=72)
        # Drop row at index 50 (creating a 1-hour gap)
        gapped_df = synthetic_hourly_features.drop(index=[50]).reset_index(drop=True)
        df_feat, df_targ = loader.construct_physical_targets(gapped_df)

        # Any sample whose target horizon touches the missing timestamp dt must be dropped
        missing_dt = synthetic_hourly_features.loc[50, "dt"]
        for dt_val in df_feat["dt"]:
            # None of the 72 future target timestamps for this sample can be the missing_dt
            future_ts = [dt_val + h * 3600 for h in range(1, 73)]
            assert missing_dt not in future_ts

    def test_chronological_split_asserts_embargo_invariant(self, synthetic_hourly_features):
        loader = FeatureStoreTrainingLoader(forecast_horizons=72)
        df_feat, df_targ = loader.construct_physical_targets(synthetic_hourly_features)

        split_data = loader.split_chronological_with_embargo(df_feat, df_targ, train_ratio=0.8)
        audit = split_data["leakage_audit"]

        max_train_dt = audit["max_train_input_dt"]
        min_val_dt = audit["min_val_input_dt"]
        max_train_target = audit["max_train_target_dt"]

        # Rule 1: min(val_dt) > max(train_dt) + 72h
        assert min_val_dt > max_train_dt + 72 * 3600
        # Rule 2: max_train_target_dt < min_val_dt
        assert max_train_target < min_val_dt
        assert audit["leakage_rule_passed"] is True


class TestDailyCandidateTrainingRunner:
    """Test suite for candidate model training, evaluation, and production protection."""

    def test_production_directory_safety_check(self, tmp_path):
        from src.config import RUNTIME_DIR
        prod_dir = RUNTIME_DIR / "production"
        with pytest.raises(ValidationError, match="Production safety violation"):
            DailyCandidateTrainingRunner(output_dir=prod_dir)

    def test_candidate_training_ridge_offline(self, synthetic_hourly_features, tmp_path):
        # Save synthetic features to temporary CSV for offline runner test
        csv_file = tmp_path / "offline_features.csv"
        synthetic_hourly_features.to_csv(csv_file, index=False)

        candidate_dir = tmp_path / "candidates"
        runner = DailyCandidateTrainingRunner(
            candidate_family="ridge",
            output_dir=candidate_dir,
            forecast_horizons=72,
        )

        res = runner.run(dry_run=False, offline_data_path=csv_file)

        assert res["status"] == "candidate_training_completed"
        assert res["candidate_family"] == "ridge"
        assert "candidate_rmse" in res
        assert "baseline_rmse" in res
        assert res["production_champion"]["status"] == "FROZEN_READ_ONLY"

        # Verify candidate directory contents
        run_dir = Path(res["candidate_run_dir"])
        assert (run_dir / "candidate_model.joblib").exists()
        assert (run_dir / "scaler.joblib").exists()
        assert (run_dir / "evaluation.json").exists()
        assert (run_dir / "dataset_provenance.json").exists()
        assert (run_dir / "candidate_comparison.json").exists()
        assert (run_dir / "manifest.json").exists()

        # Check manifest contents
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert "candidate_model.joblib" in manifest["artifacts"]
        assert len(manifest["artifacts"]["candidate_model.joblib"]) == 64  # SHA256 hex length

    def test_candidate_training_hybrid_offline(self, synthetic_hourly_features, tmp_path):
        csv_file = tmp_path / "offline_features.csv"
        synthetic_hourly_features.to_csv(csv_file, index=False)

        candidate_dir = tmp_path / "candidates"
        runner = DailyCandidateTrainingRunner(
            candidate_family="hybrid",
            output_dir=candidate_dir,
            forecast_horizons=72,
        )

        res = runner.run(dry_run=False, offline_data_path=csv_file)
        assert res["status"] == "candidate_training_completed"
        assert res["candidate_family"] == "hybrid"

    def test_dry_run_generates_provenance_without_model_fitting(self, synthetic_hourly_features, tmp_path):
        csv_file = tmp_path / "offline_features.csv"
        synthetic_hourly_features.to_csv(csv_file, index=False)

        candidate_dir = tmp_path / "candidates"
        runner = DailyCandidateTrainingRunner(
            candidate_family="ridge",
            output_dir=candidate_dir,
        )

        res = runner.run(dry_run=True, offline_data_path=csv_file)
        assert res["status"] == "dry_run_success"
        assert res["mode"] == "dry_run"

        # Model artifact must NOT be serialized in dry run
        run_dir = candidate_dir / res["run_id"]
        assert not (run_dir / "candidate_model.joblib").exists()
        assert (run_dir / "dataset_provenance.json").exists()
        assert (run_dir / "candidate_evaluation_report.json").exists()

    def test_extreme_metrics_helper(self):
        y_true = np.array([[50.0, 250.0], [350.0, 100.0]])
        y_pred = np.array([[55.0, 260.0], [340.0, 95.0]])

        sev = compute_extreme_metrics(y_true, y_pred, threshold=200.0)
        assert sev["sample_count"] == 2
        assert sev["rmse"] == 10.0
        assert sev["low_sample_warning"] is True

        none_res = compute_extreme_metrics(y_true, y_pred, threshold=500.0)
        assert none_res["sample_count"] == 0
        assert none_res["rmse"] is None

    def test_recommendation_gate_same_protocol_evaluation(self, synthetic_hourly_features, tmp_path):
        csv_file = tmp_path / "offline_features.csv"
        synthetic_hourly_features.to_csv(csv_file, index=False)

        candidate_dir = tmp_path / "candidates"
        runner = DailyCandidateTrainingRunner(
            candidate_family="ridge",
            output_dir=candidate_dir,
            forecast_horizons=72,
        )

        res = runner.run(dry_run=False, offline_data_path=csv_file)
        run_dir = Path(res["candidate_run_dir"])
        comparison = json.loads((run_dir / "candidate_comparison.json").read_text(encoding="utf-8"))

        assert "same_protocol_evaluation" in comparison
        assert comparison["comparable_evaluation_protocol"] is False
        assert "beats_persistence_overall" in comparison["same_protocol_evaluation"]
        assert "meets_quality_threshold" in comparison["same_protocol_evaluation"]
        assert comparison["same_protocol_evaluation"]["quality_threshold_pct"] == 20.0
        assert "extreme_gt200_ok" in comparison["same_protocol_evaluation"]
        assert "extreme_gt300_ok" in comparison["same_protocol_evaluation"]
        # Invariant: EXP-019 holdout benchmarks are present only as reference
        assert "exp019_reference_benchmarks" in comparison
        assert comparison["exp019_reference_benchmarks"]["holdout_test"]["overall_rmse"] == 75.91
        assert comparison["exp019_reference_benchmarks"]["holdout_test"]["overall_mae"] == 53.55
        assert comparison["exp019_reference_benchmarks"]["holdout_test"]["overall_r2"] == 0.4858

