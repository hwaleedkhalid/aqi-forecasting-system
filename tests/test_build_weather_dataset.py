"""Unit tests for build_weather_dataset pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.feature_pipeline.build_weather_dataset import run_weather_enrichment_pipeline


class TestBuildWeatherDataset:
    """Test full weather data builder flow."""

    @patch("src.data_ingestion.weather_provider.OpenMeteoWeatherProvider.fetch_historical_weather")
    def test_run_weather_enrichment_pipeline(self, mock_fetch: MagicMock, tmp_path: Path) -> None:
        n_samples = 250
        dates = pd.date_range("2025-01-01 00:00:00", periods=n_samples, freq="1h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame({
            "datetime_utc": dates,
            "temperature_2m": np.linspace(10, 25, n_samples),
            "relative_humidity_2m": np.linspace(40, 80, n_samples),
            "surface_pressure": np.linspace(1010, 1015, n_samples),
            "wind_speed_10m": np.linspace(5, 15, n_samples),
            "wind_direction_10m": np.linspace(0, 350, n_samples),
            "precipitation": np.zeros(n_samples),
        })

        # Mock clean AQI
        df_clean = pd.DataFrame({
            "datetime_utc": [str(d) for d in dates],
            "epa_aqi": np.linspace(50, 200, n_samples),
            "pm2_5": np.linspace(20, 100, n_samples),
            "pm10": np.linspace(30, 150, n_samples),
            "no2": np.linspace(10, 40, n_samples),
            "so2": np.linspace(5, 20, n_samples),
            "co": np.linspace(200, 600, n_samples),
            "o3": np.linspace(15, 60, n_samples),
            "nh3": np.linspace(1, 10, n_samples),
            "no": np.linspace(1, 10, n_samples),
        })
        clean_path = tmp_path / "historical_aqi_clean.csv"
        df_clean.to_csv(clean_path, index=False)

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        summary = run_weather_enrichment_pipeline(
            data_dir=tmp_path,
            raw_weather_dir=tmp_path / "raw_weather",
            models_dir=models_dir,
        )

        assert "version" in summary
        assert summary["version"] == "v2_weather_enriched"
        assert (tmp_path / "X_train_v2.npy").exists()
        assert (tmp_path / "X_test_v2.npy").exists()
        assert summary["train_samples"] > 0
        assert summary["test_samples"] > 0
