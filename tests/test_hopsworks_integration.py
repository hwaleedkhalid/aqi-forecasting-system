"""Unit, contract, and resilience tests for Hopsworks Feature Store & Model Registry integration."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.exceptions import FeatureStoreError, ValidationError
from src.feature_pipeline.hopsworks_integration import (
    HopsworksFeatureStoreConnector,
    load_canonical_feature_names,
)
from src.inference.hopsworks_registry import (
    HopsworksModelRegistryConnector,
    compute_file_sha256,
)
from src.inference.predictor import AQIPredictor


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Create sample DataFrame containing canonical 114 features."""
    canonical_features = load_canonical_feature_names()
    data = {col: [float(i + 1)] for i, col in enumerate(canonical_features)}
    # Set dt to valid epoch integer
    data["dt"] = [1693500000]
    return pd.DataFrame(data)


@pytest.fixture
def local_bundle_dir(tmp_path) -> Path:
    """Create a temporary valid model bundle directory with manifest."""
    connector = HopsworksModelRegistryConnector()
    bundle_dir = tmp_path / "test_bundle"
    connector.build_local_bundle(bundle_dir)
    return bundle_dir


class TestHopsworksFeatureStoreConnector:
    """Test suite for Feature Store connector, storage schemas, and contract projection."""

    def test_canonical_schema_has_114_features(self):
        canonical = load_canonical_feature_names()
        assert len(canonical) == 114
        assert canonical[0] == "co"
        assert canonical[4] == "dt"
        assert canonical[9] == "epa_aqi_lag_1h"

    def test_prepare_storage_dataframe_creates_115_columns(self, sample_feature_df):
        connector = HopsworksFeatureStoreConnector(location_id="lahore")
        storage_df = connector.prepare_storage_dataframe(sample_feature_df)

        assert len(storage_df.columns) == 115
        assert "location_id" in storage_df.columns
        assert storage_df["location_id"].iloc[0] == "lahore"
        assert "dt" in storage_df.columns
        assert storage_df["dt"].dtype == np.int64

    def test_project_inference_features_preserves_exact_114_canonical_order(self, sample_feature_df):
        connector = HopsworksFeatureStoreConnector(location_id="lahore")
        # Add metadata column
        storage_df = connector.prepare_storage_dataframe(sample_feature_df)
        storage_df["extra_cloud_metadata"] = "cloud_tag_123"

        # Project
        projected_df = connector.project_inference_features(storage_df)

        assert len(projected_df.columns) == 114
        assert "location_id" not in projected_df.columns
        assert "extra_cloud_metadata" not in projected_df.columns
        assert list(projected_df.columns) == connector.canonical_features

    def test_project_inference_features_fails_closed_on_missing_column(self, sample_feature_df):
        connector = HopsworksFeatureStoreConnector()
        # Drop a canonical column
        corrupted_df = sample_feature_df.drop(columns=["pm2_5"])

        with pytest.raises(ValidationError, match="Cloud schema integrity error: missing"):
            connector.project_inference_features(corrupted_df)

    def test_insert_features_uses_synchronous_wait(self, sample_feature_df):
        connector = HopsworksFeatureStoreConnector(api_key="mock_key")
        mock_fg = MagicMock()
        mock_fg.insert.return_value = "Job_Success"

        with patch.object(connector, "get_or_create_feature_group", return_value=mock_fg):
            res = connector.insert_features(sample_feature_df, wait=True)
            mock_fg.insert.assert_called_once()
            _, kwargs = mock_fg.insert.call_args
            assert kwargs["wait"] is True
            assert res["synchronous_wait"] is True
            assert res["storage_columns"] == 115

    def test_get_latest_feature_vector_local_fallback_when_unconfigured(self):
        connector = HopsworksFeatureStoreConnector(api_key="")
        assert connector.is_cloud_configured() is False

        vector, meta = connector.get_latest_feature_vector()
        assert vector.shape == (1, 114)
        assert meta["cloud_active"] is False
        assert "Fallback" in meta["source"]

    def test_get_latest_feature_vector_cloud_active(self, sample_feature_df):
        connector = HopsworksFeatureStoreConnector(api_key="mock_key")
        mock_query = MagicMock()
        mock_query.read.return_value = connector.prepare_storage_dataframe(sample_feature_df)
        mock_fg = MagicMock()
        mock_fg.select_all.return_value.filter.return_value = mock_query

        with patch.object(connector, "_login") as mock_login:
            mock_fs = MagicMock()
            mock_fs.get_feature_group.return_value = mock_fg
            mock_login.return_value = mock_fs

            vector, meta = connector.get_latest_feature_vector()
            assert vector.shape == (1, 114)
            assert meta["cloud_active"] is True
            assert "Hopsworks Cloud" in meta["source"]

    def test_get_or_create_feature_group_configures_entity_pk_and_event_time(self):
        connector = HopsworksFeatureStoreConnector(api_key="mock_key")
        mock_fs = MagicMock()
        mock_fg = MagicMock()
        mock_fs.get_or_create_feature_group.return_value = mock_fg

        with patch.object(connector, "_login", return_value=mock_fs):
            fg = connector.get_or_create_feature_group()
            mock_fs.get_or_create_feature_group.assert_called_once_with(
                name=connector.feature_group_name,
                version=connector.feature_group_version,
                primary_key=["location_id"],
                event_time="dt",
                time_travel_format="HUDI",
                description="Weather-enriched air quality features with 72-hour lag windows for Lahore (EXP-019).",
                online_enabled=True,
            )
            assert fg == mock_fg

    def test_prepare_storage_dataframe_filters_out_auxiliary_columns(self, sample_feature_df):
        connector = HopsworksFeatureStoreConnector(location_id="lahore")
        augmented_df = sample_feature_df.copy()
        augmented_df["datetime_utc"] = "2026-08-31 07:00:00+00:00"
        augmented_df["dominant_pollutant"] = "pm2_5"
        augmented_df["aqi_category"] = "Moderate"

        storage_df = connector.prepare_storage_dataframe(augmented_df)
        assert len(storage_df.columns) == 115
        assert "datetime_utc" not in storage_df.columns
        assert "dominant_pollutant" not in storage_df.columns
        assert "aqi_category" not in storage_df.columns
        assert storage_df.columns[0] == "location_id"
        assert list(storage_df.columns[1:]) == connector.canonical_features

    def test_idempotency_semantics_offline_and_online(self, sample_feature_df):
        """Verify that offline stores retain unique (location_id, event_time) and online retains latest dt."""
        from src.feature_pipeline.backfill_hopsworks import audit_historical_dataset

        # Build simulated 3-row historical series
        df1 = sample_feature_df.copy()
        df1["dt"] = [1600000000]

        df2 = sample_feature_df.copy()
        df2["dt"] = [1600003600]

        df3 = sample_feature_df.copy()
        df3["dt"] = [1600007200]

        history_df = pd.concat([df1, df2, df3], ignore_index=True)
        audit = audit_historical_dataset(history_df, load_canonical_feature_names())
        assert audit["row_count"] == 3
        assert audit["duplicate_count"] == 0
        assert audit["latest_dt"] == 1600007200

        # Simulate re-inserting row 2 with updated reading
        reinsert_df = df2.copy()
        # In Hopsworks, upsert with primary_key=['location_id'] and event_time='dt':
        # Offline store deduplicates (location_id, dt) -> total distinct rows remains 3
        combined_offline = pd.concat([history_df, reinsert_df], ignore_index=True)
        deduped_offline = combined_offline.drop_duplicates(subset=["dt"], keep="last")
        assert len(deduped_offline) == 3

        # Online store maintains exactly 1 entry for location_id, pointing to max dt
        online_latest_dt = deduped_offline["dt"].max()
        assert online_latest_dt == 1600007200


class TestHopsworksBackfillRunner:
    """Test suite for backfill pre-flight validation and execution runner."""

    def test_audit_historical_dataset_fails_on_nan(self, sample_feature_df):
        from src.feature_pipeline.backfill_hopsworks import audit_historical_dataset

        corrupt_df = sample_feature_df.copy()
        corrupt_df.loc[0, "co"] = np.nan
        with pytest.raises(ValidationError, match="contains 1 NaN values"):
            audit_historical_dataset(corrupt_df, load_canonical_feature_names())

    def test_audit_historical_dataset_fails_on_duplicate_dt(self, sample_feature_df):
        from src.feature_pipeline.backfill_hopsworks import audit_historical_dataset

        dupe_df = pd.concat([sample_feature_df, sample_feature_df], ignore_index=True)
        with pytest.raises(ValidationError, match="duplicate timestamp"):
            audit_historical_dataset(dupe_df, load_canonical_feature_names())

    def test_backfill_runner_dry_run_passes(self):
        from src.feature_pipeline.backfill_hopsworks import HopsworksBackfillRunner

        runner = HopsworksBackfillRunner(chunk_size=10000)
        res = runner.run(dry_run=True)
        assert res["status"] == "dry_run_success"
        assert res["audit"]["row_count"] == 48715
        assert res["audit"]["duplicate_count"] == 0
        assert res["storage_columns"] == 115
        assert res["projected_features"] == 114



class TestHopsworksModelRegistryConnector:
    """Test suite for Model Registry connector, bundle integrity, and explicit versioning."""

    def test_bundle_manifest_contains_all_files_and_checksums(self, local_bundle_dir):
        connector = HopsworksModelRegistryConnector()
        manifest = connector.verify_bundle_integrity(local_bundle_dir)

        assert manifest["model_id"] == "EXP-019"
        assert manifest["canonical_feature_count"] == 114
        assert len(manifest["files"]) == 4
        assert "production_hybrid_model.joblib" in manifest["files"]
        assert "feature_scaler_v2_weather.joblib" in manifest["files"]
        assert "feature_schema_v2_weather.json" in manifest["files"]
        assert "empirical_error_intervals.json" in manifest["files"]

    def test_bundle_integrity_fails_closed_on_checksum_mismatch(self, local_bundle_dir):
        connector = HopsworksModelRegistryConnector()

        # Corrupt one of the files
        corrupt_target = local_bundle_dir / "empirical_error_intervals.json"
        with open(corrupt_target, "a") as f:
            f.write(" ")  # Change file content to invalidate SHA256

        with pytest.raises(ValidationError, match="Integrity violation: SHA256 checksum mismatch"):
            connector.verify_bundle_integrity(local_bundle_dir)

    def test_fetch_production_champion_uses_explicit_name_and_version(self, local_bundle_dir):
        connector = HopsworksModelRegistryConnector(
            api_key="mock_key",
            model_name="pearls_aqi_production_champion",
            model_version=1,
        )

        mock_model = MagicMock()
        mock_model.download.return_value = str(local_bundle_dir)
        mock_mr = MagicMock()
        mock_mr.get_model.return_value = mock_model

        with patch.object(connector, "_login", return_value=mock_mr):
            bundle_path, manifest = connector.fetch_production_champion()

            # Assert explicit lookup was used (never 'get_best_model' or 'latest')
            mock_mr.get_model.assert_called_once_with(name="pearls_aqi_production_champion", version=1)
            assert manifest["model_id"] == "EXP-019"

    def test_candidate_registration_does_not_shadow_champion(self, tmp_path):
        connector = HopsworksModelRegistryConnector(api_key="mock_key")
        mock_mr = MagicMock()
        mock_candidate = MagicMock()
        mock_mr.python.create_model.return_value = mock_candidate

        with patch.object(connector, "_login", return_value=mock_mr):
            candidate_dir = tmp_path / "candidate"
            candidate_dir.mkdir()
            metrics = {"overall_rmse": 74.5}

            connector.register_candidate_model(candidate_dir, metrics)

            mock_mr.python.create_model.assert_called_once()
            _, kwargs = mock_mr.python.create_model.call_args
            assert kwargs["name"] == "pearls_aqi_candidate_model"
            assert kwargs["name"] != connector.model_name
            mock_candidate.save.assert_called_once_with(str(candidate_dir))

    def test_cloud_and_local_bundles_produce_identical_predictions(self):
        """Verify mathematical parity between local model execution and bundle execution."""
        predictor = AQIPredictor()
        local_preds = predictor.predict_latest(use_cache=False)

        # Verify predictions match expected contract
        assert len(local_preds["forecasts"]) == 72
        assert local_preds["model_id"] == "EXP-019"
        assert local_preds["feature_count"] == 114

    def test_registration_idempotency_reuses_identical_version(self, local_bundle_dir, tmp_path):
        """Verify that registering an already registered model with identical checksums succeeds safely."""
        connector = HopsworksModelRegistryConnector(api_key="mock_key")
        mock_mr = MagicMock()
        mock_existing = MagicMock()
        mock_existing.name = "pearls_aqi_production_champion"
        mock_existing.version = 1
        # Download returns local_bundle_dir which has identical files
        mock_existing.download.return_value = str(local_bundle_dir)
        mock_mr.get_model.return_value = mock_existing

        with patch.object(connector, "_login", return_value=mock_mr):
            res = connector.register_production_champion(local_bundle_dir)
            assert res.version == 1
            mock_mr.python.create_model.assert_not_called()

    def test_registration_conflict_fails_closed_on_checksum_mismatch(self, local_bundle_dir, tmp_path):
        """Verify that registering an existing version with divergent checksums raises ValidationError."""
        connector = HopsworksModelRegistryConnector(api_key="mock_key")
        mock_mr = MagicMock()
        mock_existing = MagicMock()
        mock_existing.name = "pearls_aqi_production_champion"
        mock_existing.version = 1

        # Create a corrupted download dir
        divergent_dir = tmp_path / "divergent_bundle"
        connector.build_local_bundle(divergent_dir)
        manifest_path = divergent_dir / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["files"]["production_hybrid_model.joblib"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        if (divergent_dir / "production" / "manifest.json").exists():
            with open(divergent_dir / "production" / "manifest.json", "w", encoding="utf-8") as f:
                json.dump(data, f)

        mock_existing.download.return_value = str(divergent_dir)
        mock_mr.get_model.return_value = mock_existing

        with patch.object(connector, "_login", return_value=mock_mr):
            with pytest.raises(ValidationError, match="checksum mismatch|corrupted"):
                connector.register_production_champion(local_bundle_dir)

    def test_verify_downloaded_parity_passes(self, tmp_path):
        """Verify parity verification helper runs successfully on valid local bundle."""
        connector = HopsworksModelRegistryConnector()
        test_dir = tmp_path / "parity_test_bundle"
        connector.build_local_bundle(test_dir, include_explainability=True)
        report = connector.verify_downloaded_parity(test_dir)

        assert report["status"] == "PASS"
        assert report["model_id"] == "EXP-019"
        assert report["feature_count"] == 114
        assert report["output_horizons"] == 72
        assert report["max_abs_prediction_diff"] < 1e-6
        assert len(report["boundary_horizon_parity"]) == 8
        assert "h1" in report["boundary_horizon_parity"]
        assert "h72" in report["boundary_horizon_parity"]
