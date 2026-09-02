"""Unit tests for OpenMeteoWeatherProvider."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.data_ingestion.weather_provider import OpenMeteoWeatherProvider
from src.exceptions import DataIngestionError, ValidationError


class TestOpenMeteoWeatherProvider:
    """Test API retrieval, schema validation, and file caching."""

    @pytest.fixture
    def mock_weather_payload(self) -> dict:
        """Sample valid Open-Meteo payload."""
        times = ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00"]
        return {
            "latitude": 31.55,
            "longitude": 74.34,
            "hourly": {
                "time": times,
                "temperature_2m": [15.2, 14.8, 14.1],
                "relative_humidity_2m": [65.0, 68.0, 70.0],
                "surface_pressure": [1012.5, 1012.3, 1012.0],
                "wind_speed_10m": [8.5, 7.2, 6.9],
                "wind_direction_10m": [120.0, 135.0, 140.0],
                "precipitation": [0.0, 0.0, 0.1],
            },
        }

    def test_parse_and_validate_payload_success(self, mock_weather_payload: dict, tmp_path: Path) -> None:
        provider = OpenMeteoWeatherProvider(raw_dir=tmp_path)
        df = provider._parse_and_validate_payload(mock_weather_payload)

        assert len(df) == 3
        assert "datetime_utc" in df.columns
        assert "temperature_2m" in df.columns
        assert df["temperature_2m"].iloc[0] == 15.2

    def test_parse_missing_key_raises_validation_error(self, tmp_path: Path) -> None:
        provider = OpenMeteoWeatherProvider(raw_dir=tmp_path)
        bad_payload = {"hourly": {"time": ["2026-01-01T00:00"]}}
        with pytest.raises(ValidationError, match="Missing expected variable"):
            provider._parse_and_validate_payload(bad_payload)

    def test_temperature_out_of_bounds_raises_error(self, mock_weather_payload: dict, tmp_path: Path) -> None:
        mock_weather_payload["hourly"]["temperature_2m"][0] = 100.0  # Invalid > 65
        provider = OpenMeteoWeatherProvider(raw_dir=tmp_path)
        with pytest.raises(ValidationError, match="Temperature out of physical atmospheric bounds"):
            provider._parse_and_validate_payload(mock_weather_payload)

    @patch("requests.get")
    def test_fetch_historical_weather_with_caching(
        self, mock_get: MagicMock, mock_weather_payload: dict, tmp_path: Path
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_weather_payload
        mock_get.return_value = mock_response

        provider = OpenMeteoWeatherProvider(raw_dir=tmp_path)
        df = provider.fetch_historical_weather(start_date="2026-01-01", end_date="2026-01-01")

        assert len(df) == 3
        cache_file = tmp_path / "openmeteo_lahore_historical.json"
        assert cache_file.exists()

        # Second call should load from cache without calling requests.get
        mock_get.reset_mock()
        df_cached = provider.fetch_historical_weather(start_date="2026-01-01", end_date="2026-01-01")
        assert len(df_cached) == 3
        mock_get.assert_not_called()
