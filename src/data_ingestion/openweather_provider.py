"""Pearls AQI Predictor - OpenWeather Data Provider.

Implements the DataProvider interface for OpenWeather APIs.
Handles current, forecast, and historical air pollution and weather retrieval
with exponential backoff retries, schema validation, and timestamp normalization.
"""

import time
from typing import Any
import requests

from src.config import (
    API_MAX_RETRIES,
    API_RETRY_BASE_DELAY,
    API_TIMEOUT_SECONDS,
    BACKFILL_START_DATE,
    OPENWEATHER_API_KEY,
    OPENWEATHER_AQ_CURRENT_URL,
    OPENWEATHER_AQ_FORECAST_URL,
    OPENWEATHER_AQ_HISTORY_URL,
    OPENWEATHER_FORECAST_URL,
    OPENWEATHER_WEATHER_URL,
    POLLUTANTS,
)
from src.data_ingestion.base_provider import DataProvider
from src.exceptions import DataIngestionError, ValidationError
from src.logger import logger

# Earliest available timestamp for OpenWeather Air Pollution History: 2020-11-27 13:00:00 UTC
MIN_HISTORY_TIMESTAMP = 1606482000


class OpenWeatherProvider(DataProvider):
    """Data provider for OpenWeather REST APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = API_TIMEOUT_SECONDS,
        max_retries: int = API_MAX_RETRIES,
        retry_base_delay: float = API_RETRY_BASE_DELAY,
    ) -> None:
        """Initialize OpenWeatherProvider.

        Args:
            api_key: OpenWeather API key. Defaults to OPENWEATHER_API_KEY from config.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for transient errors.
            retry_base_delay: Initial backoff delay in seconds.
        """
        self.api_key = api_key if api_key is not None else OPENWEATHER_API_KEY
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def fetch_current_air_quality(self, lat: float, lon: float) -> dict:
        """Fetch current air pollution data for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Validated dictionary containing current air quality data.
        """
        self._validate_coordinates(lat, lon)
        params = {"lat": lat, "lon": lon, "appid": self._get_valid_api_key()}
        data = self._make_request(OPENWEATHER_AQ_CURRENT_URL, params)
        self._validate_air_quality_response(data, expected_min_records=1)
        return data

    def fetch_air_quality_forecast(self, lat: float, lon: float) -> dict:
        """Fetch air quality forecast for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Validated dictionary containing hourly forecast data.
        """
        self._validate_coordinates(lat, lon)
        params = {"lat": lat, "lon": lon, "appid": self._get_valid_api_key()}
        data = self._make_request(OPENWEATHER_AQ_FORECAST_URL, params)
        self._validate_air_quality_response(data, expected_min_records=1)
        return data

    def fetch_historical_air_quality(
        self, lat: float, lon: float, start: int, end: int
    ) -> dict:
        """Fetch historical air quality records between start and end timestamps.

        Note: OpenWeather treats start and end timestamps inclusively.
        Records are returned in hourly intervals and sorted chronologically.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).
            start: Start Unix timestamp in UTC.
            end: End Unix timestamp in UTC.

        Returns:
            Validated dictionary containing historical hourly air quality observations.
        """
        self._validate_coordinates(lat, lon)
        self._validate_time_range(start, end)

        params = {
            "lat": lat,
            "lon": lon,
            "start": start,
            "end": end,
            "appid": self._get_valid_api_key(),
        }
        data = self._make_request(OPENWEATHER_AQ_HISTORY_URL, params)
        self._validate_air_quality_response(data, expected_min_records=0)

        # Normalize timestamps: ensure chronological order and deduplicate
        if "list" in data and isinstance(data["list"], list):
            seen_dts = set()
            unique_records = []
            for item in sorted(data["list"], key=lambda x: x.get("dt", 0)):
                dt = item.get("dt")
                if dt not in seen_dts:
                    seen_dts.add(dt)
                    unique_records.append(item)
            data["list"] = unique_records

        return data

    def fetch_current_weather(self, lat: float, lon: float) -> dict:
        """Fetch current weather data for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Validated dictionary containing current weather data.
        """
        self._validate_coordinates(lat, lon)
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._get_valid_api_key(),
            "units": "metric",
        }
        data = self._make_request(OPENWEATHER_WEATHER_URL, params)
        self._validate_weather_response(data)
        return data

    def fetch_weather_forecast(self, lat: float, lon: float) -> dict:
        """Fetch 5-day / 3-hour weather forecast for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Validated dictionary containing forecasted weather data.
        """
        self._validate_coordinates(lat, lon)
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self._get_valid_api_key(),
            "units": "metric",
        }
        data = self._make_request(OPENWEATHER_FORECAST_URL, params)
        if not isinstance(data, dict) or "list" not in data:
            raise ValidationError(
                "Invalid weather forecast response format",
                detail="Missing 'list' key in forecast response",
            )
        return data

    # =========================================================================
    # Internal Request & Validation Helpers
    # =========================================================================

    def _get_valid_api_key(self) -> str:
        """Validate and return the API key."""
        if not self.api_key or not self.api_key.strip():
            raise ValidationError(
                "OpenWeather API key is missing or empty",
                detail="Ensure OPENWEATHER_API_KEY is configured in .env or constructor",
            )
        return self.api_key.strip()

    def _validate_coordinates(self, lat: float, lon: float) -> None:
        """Validate geographic coordinates."""
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValidationError(
                "Coordinates must be numeric",
                detail=f"Received lat={lat} (type: {type(lat)}), lon={lon} (type: {type(lon)})",
            )
        if not (-90.0 <= lat <= 90.0):
            raise ValidationError(
                f"Latitude {lat} out of valid range [-90.0, 90.0]"
            )
        if not (-180.0 <= lon <= 180.0):
            raise ValidationError(
                f"Longitude {lon} out of valid range [-180.0, 180.0]"
            )

    def _validate_time_range(self, start: int, end: int) -> None:
        """Validate historical query start and end Unix timestamps."""
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValidationError(
                "Start and end timestamps must be integers",
                detail=f"Received start={start} ({type(start)}), end={end} ({type(end)})",
            )
        if start < MIN_HISTORY_TIMESTAMP:
            raise ValidationError(
                f"Start timestamp {start} is earlier than OpenWeather earliest history date ({BACKFILL_START_DATE})",
                detail=f"Minimum timestamp is {MIN_HISTORY_TIMESTAMP}",
            )
        if start > end:
            raise ValidationError(
                f"Start timestamp ({start}) cannot be greater than end timestamp ({end})"
            )

    def _make_request(self, url: str, params: dict[str, Any]) -> dict:
        """Execute HTTP GET request with exponential backoff retry logic.

        Args:
            url: API endpoint URL.
            params: Query parameters (includes API key).

        Returns:
            Parsed JSON dictionary from API response.

        Raises:
            DataIngestionError: If the request fails after maximum retries.
        """
        redacted_url = self._redact_url(url, params)
        last_exception: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()
            try:
                logger.debug(f"API Request attempt {attempt}/{self.max_retries}: {redacted_url}")
                response = requests.get(url, params=params, timeout=self.timeout)
                duration = time.time() - start_time

                # Check HTTP status
                if response.status_code == 200:
                    logger.info(
                        f"API Request successful: {redacted_url} (status={response.status_code}, duration={duration:.2f}s)"
                    )
                    return response.json()

                # Non-retryable client errors (400, 401, 403, 404)
                if response.status_code in [400, 401, 403, 404]:
                    msg = f"Client error ({response.status_code}) from OpenWeather API: {response.text}"
                    logger.error(msg)
                    raise DataIngestionError(
                        f"OpenWeather API returned status {response.status_code}",
                        detail=response.text,
                    )

                # Retryable server errors (500, 502, 503, 504) or Rate Limit (429)
                logger.warning(
                    f"Retryable status {response.status_code} received on attempt {attempt}/{self.max_retries}: {response.text}"
                )
                last_exception = DataIngestionError(
                    f"HTTP error {response.status_code}", detail=response.text
                )

            except (requests.Timeout, requests.ConnectionError) as req_err:
                duration = time.time() - start_time
                logger.warning(
                    f"Network error on attempt {attempt}/{self.max_retries}: {type(req_err).__name__} ({duration:.2f}s)"
                )
                last_exception = req_err

            # Exponential backoff before next attempt
            if attempt < self.max_retries:
                sleep_time = self.retry_base_delay * (2 ** (attempt - 1))
                logger.debug(f"Backing off for {sleep_time:.2f}s before retry...")
                time.sleep(sleep_time)

        # Max retries exhausted
        err_msg = f"Failed to fetch data from {redacted_url} after {self.max_retries} attempts"
        logger.error(err_msg)
        raise DataIngestionError(err_msg, detail=str(last_exception))

    def _validate_air_quality_response(
        self, data: Any, expected_min_records: int = 0
    ) -> None:
        """Validate structure and content of air pollution API response."""
        if not isinstance(data, dict):
            raise ValidationError(
                "Invalid response format: root must be a JSON object",
                detail=f"Received type: {type(data)}",
            )

        if "list" not in data or not isinstance(data["list"], list):
            raise ValidationError(
                "Invalid response format: missing or invalid 'list' array",
                detail=f"Keys found: {list(data.keys())}",
            )

        records = data["list"]
        if len(records) < expected_min_records:
            raise ValidationError(
                f"Expected at least {expected_min_records} records, got {len(records)}"
            )

        for i, item in enumerate(records):
            if not isinstance(item, dict):
                raise ValidationError(f"Record at index {i} is not a dictionary")

            # Check timestamp
            if "dt" not in item or not isinstance(item["dt"], int):
                raise ValidationError(f"Record at index {i} missing valid 'dt' integer timestamp")

            # Check main AQI
            if "main" not in item or not isinstance(item["main"], dict) or "aqi" not in item["main"]:
                raise ValidationError(f"Record at index {i} missing 'main.aqi'")

            aqi_val = item["main"]["aqi"]
            if not isinstance(aqi_val, int) or not (1 <= aqi_val <= 5):
                raise ValidationError(
                    f"Record at index {i} has invalid OpenWeather CAQI value {aqi_val} (expected 1-5)"
                )

            # Check pollutant components
            if "components" not in item or not isinstance(item["components"], dict):
                raise ValidationError(f"Record at index {i} missing 'components' dictionary")

            components = item["components"]
            for pollutant in POLLUTANTS:
                if pollutant not in components:
                    raise ValidationError(
                        f"Record at index {i} missing required pollutant '{pollutant}'"
                    )
                val = components[pollutant]
                if not isinstance(val, (int, float)) or val < 0:
                    raise ValidationError(
                        f"Record at index {i} pollutant '{pollutant}' has invalid concentration: {val}"
                    )

    def _validate_weather_response(self, data: Any) -> None:
        """Validate structure of current weather API response."""
        if not isinstance(data, dict):
            raise ValidationError("Invalid weather response format: root must be a dictionary")
        if "main" not in data or not isinstance(data["main"], dict):
            raise ValidationError("Weather response missing 'main' measurements dictionary")
        if "temp" not in data["main"] or "humidity" not in data["main"]:
            raise ValidationError("Weather 'main' missing required 'temp' or 'humidity'")

    def _redact_url(self, url: str, params: dict[str, Any]) -> str:
        """Build URL string with API key masked for safe logging."""
        safe_params = {
            k: (v if k != "appid" else "***REDACTED***") for k, v in params.items()
        }
        param_str = "&".join(f"{k}={v}" for k, v in safe_params.items())
        return f"{url}?{param_str}"
