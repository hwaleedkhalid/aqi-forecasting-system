"""Unit tests for WeatherFeatureEngineer."""

import numpy as np
import pandas as pd
import pytest

from src.exceptions import FeatureStoreError, ValidationError
from src.feature_pipeline.weather_features import WeatherFeatureEngineer


class TestWeatherFeatureEngineer:
    """Test feature engineering on weather time series and temporal fusion."""

    @pytest.fixture
    def sample_weather_data(self) -> pd.DataFrame:
        dates = pd.date_range("2026-01-01 00:00:00", periods=50, freq="1h", tz="UTC")
        return pd.DataFrame({
            "datetime_utc": dates,
            "temperature_2m": np.linspace(10, 25, 50),
            "relative_humidity_2m": np.linspace(40, 80, 50),
            "surface_pressure": np.linspace(1010, 1015, 50),
            "wind_speed_10m": np.linspace(5, 15, 50),
            "wind_direction_10m": np.linspace(0, 350, 50),
            "precipitation": np.zeros(50),
        })

    def test_build_weather_features_creates_expected_columns(self, sample_weather_data: pd.DataFrame) -> None:
        engineer = WeatherFeatureEngineer(lag_hours=[1, 3], rolling_windows=[6])
        df_feats = engineer.build_weather_features(sample_weather_data)

        assert "wind_dir_sin" in df_feats.columns
        assert "wind_dir_cos" in df_feats.columns
        assert "pressure_diff_1h" in df_feats.columns
        assert "temperature_2m_lag_1h" in df_feats.columns
        assert "temperature_2m_rolling_mean_6h" in df_feats.columns

    def test_fuse_features_and_schema_saving(self, sample_weather_data: pd.DataFrame, tmp_path) -> None:
        engineer = WeatherFeatureEngineer(lag_hours=[1, 2], rolling_windows=[3])
        df_weather = engineer.build_weather_features(sample_weather_data)

        df_pollutants = pd.DataFrame({
            "datetime_utc": sample_weather_data["datetime_utc"],
            "pm2_5": np.ones(50) * 100.0,
            "epa_aqi": np.ones(50) * 174.0,
        })

        fused = engineer.fuse_features(df_pollutants, df_weather, drop_na=True)
        assert len(fused) > 0
        assert "stagnation_index" in fused.columns

        schema_path = tmp_path / "schema.json"
        engineer.save_enriched_schema(output_path=schema_path)
        assert schema_path.exists()
