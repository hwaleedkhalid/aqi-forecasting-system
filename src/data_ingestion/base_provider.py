"""Pearls AQI Predictor - Abstract Data Provider.

Defines the base interface for air quality and weather data providers.
Adheres to the Open/Closed Principle to allow integrating multiple providers
(e.g., OpenWeatherProvider in Phase 1-3, AQICNProvider in Phase 18).
"""

from abc import ABC, abstractmethod


class DataProvider(ABC):
    """Abstract base class for all data providers."""

    @abstractmethod
    def fetch_current_air_quality(self, lat: float, lon: float) -> dict:
        """Fetch current air pollution data for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Dictionary containing standard air pollution measurements.

        Raises:
            DataIngestionError: If API call fails after retries.
            ValidationError: If parameters or response fail validation.
        """
        pass

    @abstractmethod
    def fetch_air_quality_forecast(self, lat: float, lon: float) -> dict:
        """Fetch upcoming air quality forecast for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Dictionary containing forecasted air pollution measurements.

        Raises:
            DataIngestionError: If API call fails after retries.
            ValidationError: If parameters or response fail validation.
        """
        pass

    @abstractmethod
    def fetch_historical_air_quality(
        self, lat: float, lon: float, start: int, end: int
    ) -> dict:
        """Fetch historical air quality records between timestamps.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).
            start: Start Unix timestamp in UTC.
            end: End Unix timestamp in UTC.

        Returns:
            Dictionary containing historical hourly air pollution observations.

        Raises:
            DataIngestionError: If API call fails after retries.
            ValidationError: If parameters or response fail validation.
        """
        pass

    @abstractmethod
    def fetch_current_weather(self, lat: float, lon: float) -> dict:
        """Fetch current weather data for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Dictionary containing current weather measurements.

        Raises:
            DataIngestionError: If API call fails after retries.
            ValidationError: If parameters or response fail validation.
        """
        pass

    @abstractmethod
    def fetch_weather_forecast(self, lat: float, lon: float) -> dict:
        """Fetch weather forecast for coordinates.

        Args:
            lat: Latitude (-90 to 90).
            lon: Longitude (-180 to 180).

        Returns:
            Dictionary containing forecasted weather measurements.

        Raises:
            DataIngestionError: If API call fails after retries.
            ValidationError: If parameters or response fail validation.
        """
        pass
