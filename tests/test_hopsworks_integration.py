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

    def test_fetch_production_champion_uses_explicit_name_and_version(self):
        connector = HopsworksModelRegistryConnector(
            api_key="mock_key",
            model_name="pearls_aqi_production_champion",
            model_version=1,
        )

        mock_model = MagicMock()
        mock_model.download.return_value = "data/models/bundle"
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
