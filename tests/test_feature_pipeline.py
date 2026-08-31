"""Unit tests for Feature Engineering Pipeline.

Validates anti-leakage time-alignment, temporal cyclical encodings, backward lag
calculations, backward rolling aggregates, zero-division ratio safety, and schema persistence.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.exceptions import ValidationError
from src.feature_pipeline.feature_engineering import (
    FeatureEngineeringPipeline,
)


# =============================================================================
# Test Fixtures & Sample Time Series
# =============================================================================

@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generate 100 hours of synthetic continuous pollutant data."""
    dates = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1h", tz="UTC")
    np.random.seed(42)

    df = pd.DataFrame({
        "datetime_utc": dates,
        "pm2_5": np.linspace(20, 120, 100) + np.sin(np.linspace(0, 10, 100)) * 10,
        "pm10": np.linspace(40, 200, 100) + np.cos(np.linspace(0, 10, 100)) * 15,
        "no2": np.random.uniform(10, 80, 100),
        "so2": np.random.uniform(5, 30, 100),
        "co": np.random.uniform(200, 1500, 100),
        "o3": np.random.uniform(10, 90, 100),
        "nh3": np.random.uniform(2, 20, 100),
        "epa_aqi": np.linspace(50, 180, 100),
    })
    return df


# =============================================================================
# Time Series Alignment Tests
# =============================================================================

class TestTimeSeriesPreparation:
    """Test DatetimeIndex preparation and short gap interpolation."""

    def test_missing_timestamp_raises_error(self) -> None:
        pipeline = FeatureEngineeringPipeline()
        df_bad = pd.DataFrame({"pm2_5": [10, 20, 30]})
        with pytest.raises(ValidationError, match="must contain 'datetime_utc' or 'dt'"):
            pipeline.prepare_time_series(df_bad)

    def test_input_with_dt_column(self) -> None:
        pipeline = FeatureEngineeringPipeline()
        df_dt = pd.DataFrame({"dt": [1606482000, 1606485600], "pm2_5": [10.0, 20.0]})
        df_prep = pipeline.prepare_time_series(df_dt)
        assert len(df_prep) == 2
        assert df_prep.index[0] == pd.Timestamp("2020-11-27 13:00:00", tz="UTC")

    def test_reindexing_creates_continuous_hourly_grid(self) -> None:
        pipeline = FeatureEngineeringPipeline()
        # Create dataset with 2-hour gap
        dts = [
            pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
            # Gap at 02:00
            pd.Timestamp("2024-01-01 03:00:00", tz="UTC"),
        ]
        df_gap = pd.DataFrame({"datetime_utc": dts, "pm2_5": [10.0, 20.0, 40.0], "epa_aqi": [50.0, 60.0, 80.0]})
        df_prep = pipeline.prepare_time_series(df_gap)

        assert len(df_prep) == 4  # 00:00, 01:00, 02:00, 03:00
        # Linearly interpolated value at 02:00 should be 30.0
        assert df_prep.loc[pd.Timestamp("2024-01-01 02:00:00", tz="UTC"), "pm2_5"] == 30.0


# =============================================================================
# Temporal Features Tests
# =============================================================================

class TestTemporalFeatures:
    """Test cyclical and calendar encodings."""

    def test_cyclical_bounds(self, sample_timeseries_df: pd.DataFrame) -> None:
        pipeline = FeatureEngineeringPipeline()
        df_prep = pipeline.prepare_time_series(sample_timeseries_df)
        df_feat = pipeline.create_temporal_features(df_prep)

        for col in ["hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"]:
            assert col in df_feat.columns
            assert df_feat[col].min() >= -1.0
            assert df_feat[col].max() <= 1.0

    def test_weekend_flag(self) -> None:
        pipeline = FeatureEngineeringPipeline()
        # Monday (2024-01-01) vs Saturday (2024-01-06)
        dts = [
            pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),  # Monday
            pd.Timestamp("2024-01-06 12:00:00", tz="UTC"),  # Saturday
        ]
        df_test = pd.DataFrame({"datetime_utc": dts, "pm2_5": [10.0, 20.0]})
        df_prep = pipeline.prepare_time_series(df_test)
        df_feat = pipeline.create_temporal_features(df_prep)

        assert df_feat.loc[dts[0], "is_weekend"] == 0
        assert df_feat.loc[dts[1], "is_weekend"] == 1


# =============================================================================
# Anti-Leakage Lag & Rolling Tests
# =============================================================================

class TestLagsAndRollingAntiLeakage:
    """Verify that lags and rolling aggregates only look backward in time."""

    def test_lag_values_match_past_observations(
        self, sample_timeseries_df: pd.DataFrame
    ) -> None:
        pipeline = FeatureEngineeringPipeline(lag_hours=[1, 6, 24])
        df_prep = pipeline.prepare_time_series(sample_timeseries_df)
        df_feat = pipeline.create_lag_features(df_prep)

        # Check at row index 30
        idx_current = df_feat.index[30]
        idx_past_1h = df_feat.index[29]
        idx_past_6h = df_feat.index[24]
        idx_past_24h = df_feat.index[6]

        assert df_feat.loc[idx_current, "pm2_5_lag_1h"] == df_prep.loc[idx_past_1h, "pm2_5"]
        assert df_feat.loc[idx_current, "pm2_5_lag_6h"] == df_prep.loc[idx_past_6h, "pm2_5"]
        assert df_feat.loc[idx_current, "pm2_5_lag_24h"] == df_prep.loc[idx_past_24h, "pm2_5"]

    def test_rolling_aggregates_include_only_past_and_current(
        self, sample_timeseries_df: pd.DataFrame
    ) -> None:
        pipeline = FeatureEngineeringPipeline(rolling_windows=[6, 24])
        df_prep = pipeline.prepare_time_series(sample_timeseries_df)
        df_feat = pipeline.create_rolling_features(df_prep)

        # Test rolling mean over 6 hours at index 10
        expected_mean = df_prep["pm2_5"].iloc[5:11].mean()  # 6 elements [5..10]
        actual_mean = df_feat["pm2_5_rolling_mean_6h"].iloc[10]
        assert np.isclose(actual_mean, expected_mean)

    def test_ratios_handle_zero_safely(self) -> None:
        pipeline = FeatureEngineeringPipeline()
        # Zero denominator in pm10 and o3
        df_zero = pd.DataFrame({
            "datetime_utc": [pd.Timestamp("2024-01-01 00:00:00", tz="UTC")],
            "pm2_5": [50.0],
            "pm10": [0.0],
            "no2": [30.0],
            "o3": [0.0],
            "co": [500.0],
            "epa_aqi": [120.0],
        })
        df_prep = pipeline.prepare_time_series(df_zero)
        df_feat = pipeline.create_ratio_and_diff_features(df_prep)

        # Should not produce inf or NaN
        assert not np.isinf(df_feat["pm_ratio"].iloc[0])
        assert not np.isnan(df_feat["pm_ratio"].iloc[0])
        assert not np.isinf(df_feat["nitrogen_ozone_ratio"].iloc[0])


# =============================================================================
# Full Pipeline & Schema Tests
# =============================================================================

class TestFullPipelineAndSchema:
    """Test end-to-end feature extraction and schema serialization."""

    def test_build_features_output_shape_and_ordering(
        self, sample_timeseries_df: pd.DataFrame
    ) -> None:
        pipeline = FeatureEngineeringPipeline()
        df_feat, feature_names = pipeline.build_features(sample_timeseries_df, drop_na=True)

        # Warmup of 24h lag drops 24 rows from 100 -> 76
        assert len(df_feat) == 76
        assert len(feature_names) >= 35
        # Column names must be sorted deterministically
        assert feature_names == sorted(feature_names)
        assert all(c in df_feat.columns for c in feature_names)

    def test_save_and_load_feature_schema(
        self, sample_timeseries_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        pipeline = FeatureEngineeringPipeline()
        _, feature_names = pipeline.build_features(sample_timeseries_df, drop_na=True)

        schema_file = tmp_path / "test_schema.json"
        saved_path = pipeline.save_feature_schema(schema_file)

        assert saved_path.exists()
        with open(saved_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        assert schema["total_features"] == len(feature_names)
        assert schema["feature_names"] == feature_names

    def test_save_schema_without_building_raises_error(self, tmp_path: Path) -> None:
        pipeline = FeatureEngineeringPipeline()
        with pytest.raises(ValidationError, match="Feature columns list is empty"):
            pipeline.save_feature_schema(tmp_path / "err.json")
