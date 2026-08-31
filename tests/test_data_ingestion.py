"""Unit tests for Data Ingestion Layer (OpenWeatherProvider).

Tests all API methods, retry mechanisms with exponential backoff, coordinate
validation, time range checks, and payload schema verification using mocked responses.
"""

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.data_ingestion.base_provider import DataProvider
from src.data_ingestion.openweather_provider import (
    MIN_HISTORY_TIMESTAMP,
    OpenWeatherProvider,
)
from src.exceptions import DataIngestionError, ValidationError


# =============================================================================
# Test Fixtures & Helpers
# =============================================================================

@pytest.fixture
def provider() -> OpenWeatherProvider:
    """Create OpenWeatherProvider instance with dummy API key and fast retries."""
    return OpenWeatherProvider(
        api_key="test_api_key_123",
        timeout=5,
        max_retries=3,
        retry_base_delay=0.01,  # Fast retries for testing
    )


@pytest.fixture
def valid_aq_response() -> dict:
    """Valid OpenWeather air pollution payload."""
    return {
        "coord": {"lon": 74.3436, "lat": 31.5497},
        "list": [
            {
                "dt": 1606482000,
                "main": {"aqi": 3},
                "components": {
                    "co": 350.2,
                    "no": 1.2,
                    "no2": 25.4,
                    "o3": 45.1,
                    "so2": 12.0,
                    "pm2_5": 32.5,
                    "pm10": 65.0,
                    "nh3": 5.4,
                },
            }
        ],
    }


# =============================================================================
# Initialization & Validation Tests
# =============================================================================

class TestProviderInitialization:
    """Test constructor and parameter validation."""

    def test_implements_data_provider_interface(self, provider: OpenWeatherProvider) -> None:
        assert isinstance(provider, DataProvider)

    def test_missing_api_key_raises_validation_error(self) -> None:
        p = OpenWeatherProvider(api_key="")
        with pytest.raises(ValidationError, match="API key is missing"):
            p.fetch_current_air_quality(31.5497, 74.3436)

    def test_custom_api_key(self) -> None:
        p = OpenWeatherProvider(api_key="custom_key")
        assert p.api_key == "custom_key"


class TestCoordinateValidation:
    """Test validation of latitude and longitude coordinates."""

    def test_invalid_latitude_out_of_range(self, provider: OpenWeatherProvider) -> None:
        with pytest.raises(ValidationError, match="Latitude 95.0 out of valid range"):
            provider.fetch_current_air_quality(95.0, 74.3436)

    def test_invalid_longitude_out_of_range(self, provider: OpenWeatherProvider) -> None:
        with pytest.raises(ValidationError, match="Longitude 200.0 out of valid range"):
            provider.fetch_current_air_quality(31.5497, 200.0)

    def test_non_numeric_coordinates(self, provider: OpenWeatherProvider) -> None:
        with pytest.raises(ValidationError, match="Coordinates must be numeric"):
            provider.fetch_current_air_quality("invalid", 74.3436)  # type: ignore


class TestTimeRangeValidation:
    """Test validation of historical timestamps."""

    def test_start_earlier_than_min_history(self, provider: OpenWeatherProvider) -> None:
        too_early = MIN_HISTORY_TIMESTAMP - 100000
        with pytest.raises(ValidationError, match="earlier than OpenWeather earliest history date"):
            provider.fetch_historical_air_quality(31.5497, 74.3436, start=too_early, end=MIN_HISTORY_TIMESTAMP + 3600)

    def test_start_greater_than_end(self, provider: OpenWeatherProvider) -> None:
        with pytest.raises(ValidationError, match="cannot be greater than end timestamp"):
            provider.fetch_historical_air_quality(
                31.5497, 74.3436, start=MIN_HISTORY_TIMESTAMP + 1000, end=MIN_HISTORY_TIMESTAMP
            )

    def test_non_integer_timestamps(self, provider: OpenWeatherProvider) -> None:
        with pytest.raises(ValidationError, match="must be integers"):
            provider.fetch_historical_air_quality(
                31.5497, 74.3436, start="2020-11-27", end=MIN_HISTORY_TIMESTAMP + 3600  # type: ignore
            )


# =============================================================================
# Successful API Methods Tests
# =============================================================================

class TestSuccessfulRetrieval:
    """Test API retrieval with valid mocked responses."""

    @patch("requests.get")
    def test_fetch_current_air_quality(
        self, mock_get: MagicMock, provider: OpenWeatherProvider, valid_aq_response: dict
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = valid_aq_response
        mock_get.return_value = mock_resp

        result = provider.fetch_current_air_quality(31.5497, 74.3436)
        assert result == valid_aq_response
        assert mock_get.call_count == 1

    @patch("requests.get")
    def test_fetch_air_quality_forecast(
        self, mock_get: MagicMock, provider: OpenWeatherProvider, valid_aq_response: dict
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = valid_aq_response
        mock_get.return_value = mock_resp

        result = provider.fetch_air_quality_forecast(31.5497, 74.3436)
        assert len(result["list"]) == 1

    @patch("requests.get")
    def test_fetch_historical_air_quality_with_inclusive_boundaries(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        multi_record_response = {
            "coord": {"lon": 74.3436, "lat": 31.5497},
            "list": [
                {
                    "dt": 1606485600,
                    "main": {"aqi": 3},
                    "components": {"co": 300, "no": 1, "no2": 20, "o3": 40, "so2": 10, "pm2_5": 30, "pm10": 60, "nh3": 5},
                },
                {
                    "dt": 1606482000,
                    "main": {"aqi": 2},
                    "components": {"co": 250, "no": 1, "no2": 15, "o3": 35, "so2": 8, "pm2_5": 20, "pm10": 45, "nh3": 4},
                },
                {
                    "dt": 1606482000,  # Duplicate boundary
                    "main": {"aqi": 2},
                    "components": {"co": 250, "no": 1, "no2": 15, "o3": 35, "so2": 8, "pm2_5": 20, "pm10": 45, "nh3": 4},
                },
            ],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = multi_record_response
        mock_get.return_value = mock_resp

        res = provider.fetch_historical_air_quality(
            31.5497, 74.3436, start=1606482000, end=1606485600
        )
        records = res["list"]
        assert len(records) == 2
        assert records[0]["dt"] == 1606482000
        assert records[1]["dt"] == 1606485600

    @patch("requests.get")
    def test_fetch_current_weather(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        weather_payload = {
            "main": {"temp": 32.5, "humidity": 60, "pressure": 1012},
            "wind": {"speed": 3.5, "deg": 180},
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = weather_payload
        mock_get.return_value = mock_resp

        res = provider.fetch_current_weather(31.5497, 74.3436)
        assert res["main"]["temp"] == 32.5

    @patch("requests.get")
    def test_fetch_weather_forecast(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        fc_payload = {"list": [{"dt": 1606482000, "main": {"temp": 30.0}}]}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fc_payload
        mock_get.return_value = mock_resp

        res = provider.fetch_weather_forecast(31.5497, 74.3436)
        assert len(res["list"]) == 1


# =============================================================================
# Retry Logic & Error Handling Tests
# =============================================================================

class TestRetryAndErrorHandling:
    """Test retry with exponential backoff and error classification."""

    @patch("requests.get")
    def test_client_error_fails_immediately_without_retry(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Invalid API key"
        mock_get.return_value = mock_resp

        with pytest.raises(DataIngestionError, match="status 401"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

        assert mock_get.call_count == 1

    @patch("requests.get")
    def test_server_error_500_retries_and_exhausts(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp

        with pytest.raises(DataIngestionError, match="Failed to fetch data"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

        assert mock_get.call_count == 3

    @patch("requests.get")
    def test_rate_limit_429_retries_and_succeeds(
        self, mock_get: MagicMock, provider: OpenWeatherProvider, valid_aq_response: dict
    ) -> None:
        mock_429 = MagicMock()
        mock_429.status_code = 429
        mock_429.text = "Too Many Requests"

        mock_200 = MagicMock()
        mock_200.status_code = 200
        mock_200.json.return_value = valid_aq_response

        mock_get.side_effect = [mock_429, mock_200]

        result = provider.fetch_current_air_quality(31.5497, 74.3436)
        assert result == valid_aq_response
        assert mock_get.call_count == 2

    @patch("requests.get")
    def test_timeout_retries_and_exhausts(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        mock_get.side_effect = requests.Timeout("Connection timed out")

        with pytest.raises(DataIngestionError, match="Failed to fetch data"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

        assert mock_get.call_count == 3


# =============================================================================
# Schema & Payload Validation Tests
# =============================================================================

class TestPayloadSchemaValidation:
    """Test detection and handling of malformed or invalid API payloads."""

    @patch("requests.get")
    def test_missing_list_key(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"coord": {"lon": 74.3436, "lat": 31.5497}}
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="missing or invalid 'list' array"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_root_not_a_dict(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = ["not", "a", "dict"]
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="root must be a JSON object"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_record_not_a_dict(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"list": ["not_a_dict"]}
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="is not a dictionary"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_empty_records_when_min_expected(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"list": []}
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="Expected at least 1 records, got 0"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_missing_dt_timestamp(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        bad_payload = {
            "list": [
                {
                    "main": {"aqi": 2},
                    "components": {"co": 300, "no": 1, "no2": 20, "o3": 40, "so2": 10, "pm2_5": 30, "pm10": 60, "nh3": 5},
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_payload
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="missing valid 'dt' integer timestamp"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_missing_main_aqi(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        bad_payload = {
            "list": [
                {
                    "dt": 1606482000,
                    "components": {"co": 300, "no": 1, "no2": 20, "o3": 40, "so2": 10, "pm2_5": 30, "pm10": 60, "nh3": 5},
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_payload
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="missing 'main.aqi'"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_missing_components_dict(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        bad_payload = {
            "list": [
                {
                    "dt": 1606482000,
                    "main": {"aqi": 2},
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_payload
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="missing 'components' dictionary"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_missing_required_pollutant(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        bad_payload = {
            "list": [
                {
                    "dt": 1606482000,
                    "main": {"aqi": 2},
                    "components": {
                        "co": 300,
                        "no": 1,
                        "no2": 20,
                        "o3": 40,
                        "so2": 10,
                        "pm10": 60,
                        "nh3": 5,
                    },
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_payload
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="missing required pollutant 'pm2_5'"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_invalid_caqi_integer(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        bad_payload = {
            "list": [
                {
                    "dt": 1606482000,
                    "main": {"aqi": 6},
                    "components": {
                        "co": 300,
                        "no": 1,
                        "no2": 20,
                        "o3": 40,
                        "so2": 10,
                        "pm2_5": 30,
                        "pm10": 60,
                        "nh3": 5,
                    },
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_payload
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="invalid OpenWeather CAQI value 6"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_negative_pollutant_concentration(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        bad_payload = {
            "list": [
                {
                    "dt": 1606482000,
                    "main": {"aqi": 2},
                    "components": {
                        "co": 300,
                        "no": 1,
                        "no2": 20,
                        "o3": 40,
                        "so2": 10,
                        "pm2_5": -5.0,
                        "pm10": 60,
                        "nh3": 5,
                    },
                }
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = bad_payload
        mock_get.return_value = mock_resp

        with pytest.raises(ValidationError, match="invalid concentration"):
            provider.fetch_current_air_quality(31.5497, 74.3436)

    @patch("requests.get")
    def test_invalid_weather_payloads(
        self, mock_get: MagicMock, provider: OpenWeatherProvider
    ) -> None:
        # Not a dict
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = "invalid"
        mock_get.return_value = mock_resp
        with pytest.raises(ValidationError, match="root must be a dictionary"):
            provider.fetch_current_weather(31.5497, 74.3436)

        # Missing main
        mock_resp.json.return_value = {"weather": []}
        with pytest.raises(ValidationError, match="missing 'main'"):
            provider.fetch_current_weather(31.5497, 74.3436)

        # Missing temp/humidity
        mock_resp.json.return_value = {"main": {"pressure": 1012}}
        with pytest.raises(ValidationError, match="missing required 'temp' or 'humidity'"):
            provider.fetch_current_weather(31.5497, 74.3436)

        # Weather forecast missing list
        mock_resp.json.return_value = {"city": {}}
        with pytest.raises(ValidationError, match="Missing 'list' key"):
            provider.fetch_weather_forecast(31.5497, 74.3436)
