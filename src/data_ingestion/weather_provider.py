"""Pearls AQI Predictor - Open-Meteo Historical Weather Provider.

Fetches and caches hourly historical meteorological observations (temperature,
relative humidity, surface pressure, wind speed, wind direction, and precipitation)
for atmospheric dispersion and AQI forecasting.
"""

import json
from pathlib import Path
from typing import Any
import pandas as pd
import requests

from src.config import (
    API_MAX_RETRIES,
    API_RETRY_BASE_DELAY,
    API_TIMEOUT_SECONDS,
    RAW_WEATHER_DIR,
    TARGET_LAT,
    TARGET_LON,
)
from src.exceptions import DataIngestionError, ValidationError
from src.logger import logger


class OpenMeteoWeatherProvider:
    """Provider for historical and current weather data from Open-Meteo API."""

    ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    HOURLY_VARIABLES: list[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
    ]

    def __init__(
        self,
        lat: float = TARGET_LAT,
        lon: float = TARGET_LON,
        raw_dir: Path = RAW_WEATHER_DIR,
        timeout: int = API_TIMEOUT_SECONDS,
        max_retries: int = API_MAX_RETRIES,
    ) -> None:
        """Initialize OpenMeteoWeatherProvider.

        Args:
            lat: Target latitude coordinate.
            lon: Target longitude coordinate.
            raw_dir: Destination directory for raw weather JSON artifacts.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum exponential retry attempts.
        """
        self.lat = lat
        self.lon = lon
        self.raw_dir = raw_dir
        self.timeout = timeout
        self.max_retries = max_retries

    def fetch_historical_weather(
        self,
        start_date: str = "2020-11-27",
        end_date: str = "2026-08-31",
        cache_filename: str = "openmeteo_lahore_historical.json",
        force_download: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical hourly weather observations and cache raw payload.

        Args:
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).
            cache_filename: File name for saving raw API response.
            force_download: If True, ignores cached JSON and re-queries API.

        Returns:
            DataFrame containing hourly UTC weather records with normalized columns.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.raw_dir / cache_filename

        # Load from disk cache if present and valid
        if cache_path.exists() and not force_download:
            logger.info(f"Loading cached historical weather from {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._parse_and_validate_payload(data)

        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": self.HOURLY_VARIABLES,
            "timezone": "UTC",
        }

        logger.info(
            f"Fetching historical weather from Open-Meteo ({start_date} to {end_date}, "
            f"lat={self.lat}, lon={self.lon})..."
        )

        import time
        response = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(
                    self.ARCHIVE_URL,
                    params=params,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    break
                logger.warning(
                    f"Open-Meteo returned status {response.status_code} (attempt {attempt}/{self.max_retries})"
                )
            except requests.RequestException as err:
                logger.warning(
                    f"Open-Meteo request failed on attempt {attempt}/{self.max_retries}: {err}"
                )

            if attempt < self.max_retries:
                time.sleep(API_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        if response is None or response.status_code != 200:
            err_msg = response.text if response else "No response received"
            raise DataIngestionError(
                f"Failed to fetch historical weather from Open-Meteo after {self.max_retries} attempts.",
                detail=err_msg,
            )

        data = response.json()

        # Save atomic raw cache
        tmp_path = cache_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(cache_path)
        logger.info(f"Cached raw historical weather payload to {cache_path}")

        return self._parse_and_validate_payload(data)

    def _parse_and_validate_payload(self, data: dict[str, Any]) -> pd.DataFrame:
        """Parse raw Open-Meteo response into a structured, validated DataFrame.

        Args:
            data: Raw JSON payload dictionary.

        Returns:
            Validated hourly DataFrame with UTC timestamps.
        """
        if "hourly" not in data or "time" not in data["hourly"]:
            raise ValidationError(
                "Invalid Open-Meteo payload schema: missing 'hourly' or 'hourly.time'",
                detail=str(list(data.keys())),
            )

        hourly = data["hourly"]
        times = hourly["time"]

        df_dict = {
            "datetime_utc": pd.to_datetime(times, utc=True),
        }

        for var in self.HOURLY_VARIABLES:
            if var not in hourly:
                raise ValidationError(f"Missing expected variable '{var}' in Open-Meteo response.")
            df_dict[var] = hourly[var]

        df = pd.DataFrame(df_dict).sort_values("datetime_utc").reset_index(drop=True)

        # Basic range validation
        if df["temperature_2m"].min() < -50 or df["temperature_2m"].max() > 65:
            raise ValidationError("Temperature out of physical atmospheric bounds (-50°C to 65°C)")
        if df["relative_humidity_2m"].min() < 0 or df["relative_humidity_2m"].max() > 100:
            raise ValidationError("Relative humidity out of bounds (0% to 100%)")
        if df["wind_speed_10m"].min() < 0:
            raise ValidationError("Wind speed cannot be negative.")

        logger.info(
            f"Parsed and validated {len(df)} hourly weather records from {df['datetime_utc'].min()} "
            f"to {df['datetime_utc'].max()}"
        )
        return df
