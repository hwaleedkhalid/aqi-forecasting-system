"""Unit tests for Multi-Output Training Dataset Builder.

Validates multi-horizon target construction, anti-leakage embargo gap splitting,
train-only feature scaling, and dataset artifact serialization.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.exceptions import ValidationError
from src.training_pipeline.dataset_builder import DatasetBuilder


@pytest.fixture
def sample_feature_df() -> tuple[pd.DataFrame, list[str]]:
    """Create 200 hours of continuous synthetic feature data."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=200, freq="1h", tz="UTC")
    np.random.seed(42)

    df = pd.DataFrame(
        {
            "feature_1": np.linspace(10, 50, 200),
            "feature_2": np.sin(np.linspace(0, 20, 200)) * 5,
            "epa_aqi": np.linspace(50, 250, 200),
        },
        index=dates,
    )
    feature_names = ["feature_1", "feature_2", "epa_aqi"]
    return df, feature_names


# =============================================================================
# Multi-Output Target Construction Tests
# =============================================================================

class TestTargetConstruction:
    """Test multi-horizon target shifting and alignment."""

    def test_missing_target_col_raises_error(
        self, sample_feature_df: tuple[pd.DataFrame, list[str]]
    ) -> None:
        df, features = sample_feature_df
        builder = DatasetBuilder(target_col="nonexistent_target")
        with pytest.raises(ValidationError, match="Target column 'nonexistent_target' not found"):
            builder.construct_multi_output_targets(df, features)

    def test_target_values_match_future_horizons(
        self, sample_feature_df: tuple[pd.DataFrame, list[str]]
    ) -> None:
        df, features = sample_feature_df
        builder = DatasetBuilder(forecast_horizons=10, target_col="epa_aqi")
        X, Y, timestamps = builder.construct_multi_output_targets(df, features)

        # 200 samples minus 10 tail samples = 190 samples
        assert len(X) == 190
        assert len(Y) == 190
        assert Y.shape[1] == 10

        # Check horizon values at index 0 (t=0)
        # target_h1 should equal epa_aqi at index 1 (t+1)
        # target_h10 should equal epa_aqi at index 10 (t+10)
        assert Y.loc[timestamps[0], "target_h1"] == df["epa_aqi"].iloc[1]
        assert Y.loc[timestamps[0], "target_h10"] == df["epa_aqi"].iloc[10]


# =============================================================================
# Chronological Split & Embargo Gap Tests
# =============================================================================

class TestChronologicalSplitAndEmbargo:
    """Test train/test splitting and embargo gap anti-leakage isolation."""

    def test_embargo_gap_prevents_target_overlap(
        self, sample_feature_df: tuple[pd.DataFrame, list[str]]
    ) -> None:
        df, features = sample_feature_df
        builder = DatasetBuilder(forecast_horizons=12, train_ratio=0.8, target_col="epa_aqi")
        X, Y, timestamps = builder.construct_multi_output_targets(df, features)

        split_dict = builder.split_train_test_chronological(
            X, Y, timestamps, embargo_hours=12
        )

        train_times = split_dict["train_timestamps"]
        test_times = split_dict["test_timestamps"]

        # Train end must be at least 12 hours before test start
        max_train_time = train_times.max()
        min_test_time = test_times.min()

        hours_gap = (min_test_time - max_train_time) / pd.Timedelta(hours=1)
        assert hours_gap >= 12  # Embargo gap strictly respected
        assert len(set(train_times).intersection(set(test_times))) == 0


# =============================================================================
# Scaling & Artifact Saving Tests
# =============================================================================

class TestScalingAndArtifacts:
    """Test train-only scaling and artifact persistence."""

    def test_scaler_fit_exclusively_on_train(
        self, sample_feature_df: tuple[pd.DataFrame, list[str]], tmp_path: Path
    ) -> None:
        df, features = sample_feature_df
        builder = DatasetBuilder(
            forecast_horizons=6,
            output_dir=tmp_path / "processed",
            models_dir=tmp_path / "models",
        )
        X, Y, timestamps = builder.construct_multi_output_targets(df, features)
        split_dict = builder.split_train_test_chronological(X, Y, timestamps, embargo_hours=6)

        X_train_scaled, X_test_scaled, scaler = builder.fit_and_apply_scaler(
            split_dict["X_train_raw"], split_dict["X_test_raw"], features, save_scaler=True
        )

        # Scaler mean must match X_train_raw mean exactly
        expected_train_means = split_dict["X_train_raw"][features].mean().values
        np.testing.assert_allclose(scaler.mean_, expected_train_means, rtol=1e-4)

        # X_train scaled must have mean ~ 0 and std ~ 1
        np.testing.assert_allclose(X_train_scaled.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(X_train_scaled.std(axis=0), 1.0, atol=1e-5)

        # Scaler artifact exists
        assert (tmp_path / "models" / "feature_scaler.joblib").exists()

    def test_build_and_save_dataset_artifacts(
        self, sample_feature_df: tuple[pd.DataFrame, list[str]], tmp_path: Path
    ) -> None:
        df, features = sample_feature_df
        builder = DatasetBuilder(
            forecast_horizons=6,
            train_ratio=0.8,
            output_dir=tmp_path / "processed",
            models_dir=tmp_path / "models",
        )

        summary = builder.build_and_save_dataset(df, features)

        # Verify all binary and metadata artifacts
        assert (tmp_path / "processed" / "X_train.npy").exists()
        assert (tmp_path / "processed" / "y_train.npy").exists()
        assert (tmp_path / "processed" / "X_test.npy").exists()
        assert (tmp_path / "processed" / "y_test.npy").exists()
        assert (tmp_path / "processed" / "train_timestamps.csv").exists()
        assert (tmp_path / "processed" / "test_timestamps.csv").exists()
        assert (tmp_path / "processed" / "dataset_summary.json").exists()

        assert summary["y_train_shape"][1] == 6
        assert summary["y_test_shape"][1] == 6
