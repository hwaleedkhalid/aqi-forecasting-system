"""Data Ingestion Package.

Provides abstract data provider interfaces and concrete client implementations
for fetching meteorological and air pollution observations.
"""

from src.data_ingestion.base_provider import DataProvider
from src.data_ingestion.openweather_provider import OpenWeatherProvider

__all__ = ["DataProvider", "OpenWeatherProvider"]
