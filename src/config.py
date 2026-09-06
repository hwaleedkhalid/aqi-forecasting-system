"""Pearls AQI Predictor - Configuration Module.

Loads configuration from environment variables (.env file for local dev,
GitHub Secrets for CI/CD). All configurable values are centralized here
to avoid hardcoding throughout the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists (local development)
load_dotenv()

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_AQ_DIR = RAW_DATA_DIR / "air_quality"
RAW_WEATHER_DIR = RAW_DATA_DIR / "weather"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
LOGS_DIR = DATA_DIR / "logs"

# =============================================================================
# API Configuration
# =============================================================================
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5"
OPENWEATHER_AQ_HISTORY_URL: str = f"{OPENWEATHER_BASE_URL}/air_pollution/history"
OPENWEATHER_AQ_CURRENT_URL: str = f"{OPENWEATHER_BASE_URL}/air_pollution"
OPENWEATHER_AQ_FORECAST_URL: str = f"{OPENWEATHER_BASE_URL}/air_pollution/forecast"
OPENWEATHER_WEATHER_URL: str = f"{OPENWEATHER_BASE_URL}/weather"
OPENWEATHER_FORECAST_URL: str = f"{OPENWEATHER_BASE_URL}/forecast"

# =============================================================================
# Target City
# =============================================================================
TARGET_CITY_NAME: str = os.getenv("TARGET_CITY_NAME", "Lahore")
TARGET_LAT: float = float(os.getenv("TARGET_LAT", "31.5497"))
TARGET_LON: float = float(os.getenv("TARGET_LON", "74.3436"))

# =============================================================================
# API Request Settings
# =============================================================================
API_TIMEOUT_SECONDS: int = 30
API_MAX_RETRIES: int = 3
API_RETRY_BASE_DELAY: float = 1.0  # seconds, exponential backoff
API_RATE_LIMIT_DELAY: float = 1.0  # seconds between batch requests

# =============================================================================
# Hopsworks Configuration (Phase 16)
# =============================================================================
HOPSWORKS_API_KEY: str = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT: str = os.getenv("HOPSWORKS_PROJECT", os.getenv("HOPSWORKS_PROJECT_NAME", "pearls_aqi_predictor"))
HOPSWORKS_HOST: str = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
HOPSWORKS_FEATURE_GROUP_NAME: str = "aqi_weather_features_v2"
HOPSWORKS_FEATURE_GROUP_VERSION: int = 1
HOPSWORKS_FEATURE_VIEW_NAME: str = "aqi_forecast_view"
HOPSWORKS_FEATURE_VIEW_VERSION: int = 1
HOPSWORKS_MODEL_NAME: str = "pearls_aqi_production_champion"
HOPSWORKS_MODEL_VERSION: int = 1
HOPSWORKS_CANDIDATE_MODEL_NAME: str = "pearls_aqi_candidate_model"

# =============================================================================
# Feature Engineering
# =============================================================================
LAG_HOURS: list[int] = [1, 3, 6, 12, 24]
ROLLING_WINDOWS: list[int] = [6, 12, 24]  # hours
FORECAST_HORIZONS: int = 72  # predict next 72 hours

# Key pollutants from OpenWeather API
POLLUTANTS: list[str] = ["pm2_5", "pm10", "no2", "so2", "co", "o3", "nh3"]

# =============================================================================
# Model Training
# =============================================================================
TRAIN_TEST_SPLIT_RATIO: float = 0.8  # 80% train, 20% test (time-based)
RANDOM_STATE: int = 42

# Ridge Regression
RIDGE_ALPHA: float = 1.0

# Random Forest
RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int | None = None

# TensorFlow
TF_EPOCHS: int = 100
TF_BATCH_SIZE: int = 64
TF_LEARNING_RATE: float = 0.001
TF_EARLY_STOPPING_PATIENCE: int = 10
TF_HIDDEN_LAYERS: list[dict] = [
    {"units": 128, "activation": "relu", "dropout": 0.3},
    {"units": 64, "activation": "relu", "dropout": 0.2},
    {"units": 32, "activation": "relu", "dropout": 0.0},
]

# =============================================================================
# Historical Data Backfill
# =============================================================================
# OpenWeather Air Pollution History API: data available from Nov 27, 2020
BACKFILL_START_DATE: str = "2020-11-27"

# Dataset completeness threshold (Rules.md §14)
MIN_COMPLETENESS_RATIO: float = 0.95  # 95%

# =============================================================================
# AQI Configuration
# =============================================================================
AQI_MIN: int = 0
AQI_MAX: int = 500

# EPA AQI Categories
AQI_CATEGORIES: dict[str, dict] = {
    "Good":                          {"min": 0,   "max": 50,  "color": "#00E400"},
    "Moderate":                      {"min": 51,  "max": 100, "color": "#FFFF00"},
    "Unhealthy for Sensitive Groups": {"min": 101, "max": 150, "color": "#FF7E00"},
    "Unhealthy":                     {"min": 151, "max": 200, "color": "#FF0000"},
    "Very Unhealthy":                {"min": 201, "max": 300, "color": "#8F3F97"},
    "Hazardous":                     {"min": 301, "max": 500, "color": "#7E0023"},
}

# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE: str = "pearls_aqi.log"

# =============================================================================
# Flask API
# =============================================================================
FLASK_HOST: str = "0.0.0.0"
FLASK_PORT: int = 5000
FLASK_DEBUG: bool = False

# =============================================================================
# Streamlit Dashboard
# =============================================================================
STREAMLIT_PORT: int = 8501
