"""Shared test fixtures and configuration for pytest."""

import pytest


@pytest.fixture
def sample_pollutant_data() -> dict:
    """Return a sample OpenWeather air pollution API response."""
    return {
        "coord": {"lon": 74.3436, "lat": 31.5497},
        "list": [
            {
                "dt": 1606482000,
                "main": {"aqi": 4},
                "components": {
                    "co": 1120.44,
                    "no": 2.35,
                    "no2": 45.82,
                    "o3": 18.67,
                    "so2": 22.15,
                    "pm2_5": 78.92,
                    "pm10": 112.34,
                    "nh3": 12.56,
                },
            }
        ],
    }
