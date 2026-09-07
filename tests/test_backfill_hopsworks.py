"""Tests for Hopsworks historical feature backfill pipeline and audit rules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.exceptions import FeatureStoreError, ValidationError
from src.feature_pipeline.backfill_hopsworks import (
    HopsworksBackfillRunner,
    audit_historical_dataset,
    chunk_dataframe,
)


@pytest.fixture
def canonical_schema_114() -> list[str]:
    """Generate canonical 114 schema column list."""
    base_features = ["dt"]
    for i in range(1, 114):
        base_features.append(f"feature_{i}")
    return base_features


@pytest.fixture
def sample_valid_df(canonical_schema_114: list[str]) -> pd.DataFrame:
    """Generate valid sample DataFrame with integral dt and 114 features."""
    rows = []
    # 5 hourly consecutive timestamps
    base_dt = 1606568400
    for idx in range(5):
        row_dict = {col: 1.23 * (idx + 1) for col in canonical_schema_114}
        row_dict["dt"] = base_dt + idx * 3600
        rows.append(row_dict)
    return pd.DataFrame(rows)


class TestBackfillAudit:
    """Audit function unit tests."""

    def test_audit_valid_dataset_passes(
        self, sample_valid_df: pd.DataFrame, canonical_schema_114: list[str]
    ):
        report = audit_historical_dataset(sample_valid_df, canonical_schema_114)
        assert report["row_count"] == 5
        assert report["canonical_feature_count"] == 114
        assert report["earliest_dt"] == 1606568400
        assert report["latest_dt"] == 1606568400 + 4 * 3600
        assert report["missing_gap_hours"] == 0
        assert report["completeness_percentage"] == 100.0
        assert report["nan_count"] == 0
        assert report["inf_count"] == 0
        assert report["duplicate_count"] == 0

    def test_audit_empty_dataframe_raises(self, canonical_schema_114: list[str]):
        with pytest.raises(ValidationError, match="Historical dataset is empty"):
            audit_historical_dataset(pd.DataFrame(), canonical_schema_114)

    def test_audit_missing_canonical_feature_raises(
        self, sample_valid_df: pd.DataFrame, canonical_schema_114: list[str]
    ):
        invalid_df = sample_valid_df.drop(columns=["feature_1"])
        with pytest.raises(ValidationError, match="missing 1 canonical features"):
            audit_historical_dataset(invalid_df, canonical_schema_114)

    def test_audit_missing_dt_raises(
        self, sample_valid_df: pd.DataFrame, canonical_schema_114: list[str]
    ):
        invalid_df = sample_valid_df.drop(columns=["dt"])
        canonical_without_dt = [c for c in canonical_schema_114 if c != "dt"]
        with pytest.raises(ValidationError, match="missing event_time column 'dt'"):
            audit_historical_dataset(invalid_df, canonical_without_dt)

    def test_audit_fractional_dt_raises(
        self, sample_valid_df: pd.DataFrame, canonical_schema_114: list[str]
    ):
        invalid_df = sample_valid_df.copy()
        invalid_df["dt"] = invalid_df["dt"].astype(float)
        invalid_df.loc[0, "dt"] = 1606568400.75
        with pytest.raises(ValidationError, match="Fractional.*timestamps detected"):
            audit_historical_dataset(invalid_df, canonical_schema_114)

    def test_audit_duplicate_dt_raises(
        self, sample_valid_df: pd.DataFrame, canonical_schema_114: list[str]
    ):
        invalid_df = sample_valid_df.copy()
        invalid_df.loc[1, "dt"] = invalid_df.loc[0, "dt"]
        with pytest.raises(ValidationError, match="contains 1 duplicate timestamp records"):
            audit_historical_dataset(invalid_df, canonical_schema_114)

    def test_audit_nan_values_raises(
        self, sample_valid_df: pd.DataFrame, canonical_schema_114: list[str]
    ):
        invalid_df = sample_valid_df.copy()
        invalid_df.loc[2, "feature_10"] = np.nan
        with pytest.raises(ValidationError, match="contains 1 NaN values"):
            audit_historical_dataset(invalid_df, canonical_schema_114)

    def test_audit_inf_values_raises(
        self, sample_valid_df: pd.DataFrame, canonical_schema_114: list[str]
    ):
        invalid_df = sample_valid_df.copy()
        invalid_df.loc[2, "feature_10"] = np.inf
        with pytest.raises(ValidationError, match="contains 1 infinite values"):
            audit_historical_dataset(invalid_df, canonical_schema_114)


class TestChunking:
    """Chunking function unit tests."""

    def test_chunk_dataframe_splits_correctly(self):
        df = pd.DataFrame({"a": range(105)})
        chunks = chunk_dataframe(df, chunk_size=50)
        assert len(chunks) == 3
        assert len(chunks[0]) == 50
        assert len(chunks[1]) == 50
        assert len(chunks[2]) == 5

    def test_chunk_dataframe_invalid_size_raises(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_dataframe(df, chunk_size=0)


class TestHopsworksBackfillRunner:
    """Backfill runner unit tests."""

    def test_dry_run_success(self, tmp_path: Path):
        # Create minimal valid CSV
        from src.feature_pipeline.hopsworks_integration import load_canonical_feature_names

        canonical = load_canonical_feature_names()
        row = {col: 1.0 for col in canonical}
        row["dt"] = 1606568400
        df = pd.DataFrame([row])
        csv_file = tmp_path / "test_features.csv"
        df.to_csv(csv_file, index=False)

        runner = HopsworksBackfillRunner(csv_path=csv_file, storage="offline")
        res = runner.run(dry_run=True)
        assert res["status"] == "dry_run_success"
        assert res["audit"]["row_count"] == 1
        assert res["storage_columns"] == 115
        assert res["projected_features"] == 114
        assert res["storage_mode"] == "offline"

    @patch("src.feature_pipeline.backfill_hopsworks.wait_for_materialization_idle")
    def test_mocked_live_run_offline_with_online_sync(
        self, mock_wait_idle: MagicMock, tmp_path: Path
    ):
        from src.feature_pipeline.hopsworks_integration import load_canonical_feature_names

        canonical = load_canonical_feature_names()
        rows = []
        for i in range(2):
            r = {col: 2.5 + i for col in canonical}
            r["dt"] = 1606568400 + i * 3600
            rows.append(r)
        df = pd.DataFrame(rows)
        csv_file = tmp_path / "test_features.csv"
        df.to_csv(csv_file, index=False)

        expected_latest_vector = df.iloc[[-1]][canonical].values.astype(np.float64)

        mock_connector = MagicMock()
        mock_connector.is_cloud_configured.return_value = True
        mock_connector.feature_group_name = "aqi_weather_features_v2"
        mock_connector.feature_group_version = 1
        mock_connector.location_id = "lahore"
        mock_connector.insert_features.return_value = {"status": "inserted", "job_result": "Job(1)"}
        mock_connector.project_inference_features.side_effect = lambda d: d[canonical]
        mock_connector.get_latest_feature_vector.return_value = (
            expected_latest_vector,
            {"dt": 1606568400 + 3600, "cloud_active": True},
        )

        mock_fg = MagicMock()
        mock_connector.get_or_create_feature_group.return_value = mock_fg

        # Mock remote offline query read
        mock_fs = MagicMock()
        mock_connector._login.return_value = mock_fs
        mock_fs.get_feature_group.return_value = mock_fg
        mock_query = MagicMock()
        mock_fg.select_all.return_value.filter.return_value = mock_query
        mock_query.read.return_value = df.copy()

        runner = HopsworksBackfillRunner(
            connector=mock_connector,
            csv_path=csv_file,
            storage="offline",
        )
        res = runner.run(dry_run=False, verify_after=True, sync_online=True)

        assert res["status"] == "completed"
        assert res["total_rows_ingested"] == 2
        assert res["storage_mode"] == "offline"
        assert res["online_sync"]["status"] == "synchronized"
        assert res["online_sync"]["latest_dt"] == 1606568400 + 3600
        assert res["verification"]["online_verified"] is True
        assert res["verification"]["parity_pass"] is True
