"""Pearls AQI Predictor - REST API Package.

Provides Flask REST API serving current AQI observations, 72-hour multi-horizon
forecasts, model metadata, and health checks.
"""

from src.api.app import create_app

__all__ = ["create_app"]
