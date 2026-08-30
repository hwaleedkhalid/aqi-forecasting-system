"""Smoke tests for Phase 1 — verify project setup and configuration."""

import os
import sys
from pathlib import Path


# Ensure the project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestFolderStructure:
    """Verify that all required directories exist."""

    def test_data_raw_air_quality_exists(self) -> None:
        assert (PROJECT_ROOT / "data" / "raw" / "air_quality").is_dir()

    def test_data_raw_weather_exists(self) -> None:
        assert (PROJECT_ROOT / "data" / "raw" / "weather").is_dir()

    def test_data_processed_exists(self) -> None:
        assert (PROJECT_ROOT / "data" / "processed").is_dir()

    def test_data_models_exists(self) -> None:
        assert (PROJECT_ROOT / "data" / "models").is_dir()

    def test_data_predictions_exists(self) -> None:
        assert (PROJECT_ROOT / "data" / "predictions").is_dir()

    def test_data_logs_exists(self) -> None:
        assert (PROJECT_ROOT / "data" / "logs").is_dir()

    def test_notebooks_exists(self) -> None:
        assert (PROJECT_ROOT / "notebooks").is_dir()

    def test_src_exists(self) -> None:
        assert (PROJECT_ROOT / "src").is_dir()

    def test_tests_exists(self) -> None:
        assert (PROJECT_ROOT / "tests").is_dir()

    def test_github_workflows_exists(self) -> None:
        assert (PROJECT_ROOT / ".github" / "workflows").is_dir()


class TestConfig:
    """Verify that configuration loads correctly."""

    def test_config_imports(self) -> None:
        from src import config
        assert config is not None

    def test_project_root_is_correct(self) -> None:
        from src.config import PROJECT_ROOT as config_root
        assert config_root == PROJECT_ROOT

    def test_openweather_base_url(self) -> None:
        from src.config import OPENWEATHER_BASE_URL
        assert "openweathermap.org" in OPENWEATHER_BASE_URL

    def test_forecast_horizons(self) -> None:
        from src.config import FORECAST_HORIZONS
        assert FORECAST_HORIZONS == 72

    def test_pollutants_list(self) -> None:
        from src.config import POLLUTANTS
        assert "pm2_5" in POLLUTANTS
        assert "pm10" in POLLUTANTS
        assert "o3" in POLLUTANTS
        assert len(POLLUTANTS) == 7

    def test_aqi_categories(self) -> None:
        from src.config import AQI_CATEGORIES
        assert "Good" in AQI_CATEGORIES
        assert "Hazardous" in AQI_CATEGORIES
        assert len(AQI_CATEGORIES) == 6

    def test_aqi_range(self) -> None:
        from src.config import AQI_MIN, AQI_MAX
        assert AQI_MIN == 0
        assert AQI_MAX == 500

    def test_train_test_split(self) -> None:
        from src.config import TRAIN_TEST_SPLIT_RATIO
        assert 0 < TRAIN_TEST_SPLIT_RATIO < 1

    def test_tf_hidden_layers(self) -> None:
        from src.config import TF_HIDDEN_LAYERS
        assert len(TF_HIDDEN_LAYERS) == 3
        assert TF_HIDDEN_LAYERS[0]["units"] == 128
        assert TF_HIDDEN_LAYERS[-1]["units"] == 32

    def test_lag_hours(self) -> None:
        from src.config import LAG_HOURS
        assert LAG_HOURS == [1, 3, 6, 12, 24]

    def test_rolling_windows(self) -> None:
        from src.config import ROLLING_WINDOWS
        assert ROLLING_WINDOWS == [6, 12, 24]

    def test_env_example_exists(self) -> None:
        assert (PROJECT_ROOT / ".env.example").is_file()


class TestExceptions:
    """Verify that custom exceptions work correctly."""

    def test_base_exception(self) -> None:
        from src.exceptions import PearlsAQIError
        err = PearlsAQIError("test message", detail="extra info")
        assert "test message" in str(err)
        assert "extra info" in str(err)

    def test_data_ingestion_error(self) -> None:
        from src.exceptions import DataIngestionError, PearlsAQIError
        err = DataIngestionError("API failed")
        assert isinstance(err, PearlsAQIError)
        assert "API failed" in str(err)

    def test_feature_store_error(self) -> None:
        from src.exceptions import FeatureStoreError, PearlsAQIError
        assert issubclass(FeatureStoreError, PearlsAQIError)

    def test_model_training_error(self) -> None:
        from src.exceptions import ModelTrainingError, PearlsAQIError
        assert issubclass(ModelTrainingError, PearlsAQIError)

    def test_prediction_error(self) -> None:
        from src.exceptions import PredictionError, PearlsAQIError
        assert issubclass(PredictionError, PearlsAQIError)

    def test_validation_error(self) -> None:
        from src.exceptions import ValidationError, PearlsAQIError
        assert issubclass(ValidationError, PearlsAQIError)

    def test_exception_without_detail(self) -> None:
        from src.exceptions import PearlsAQIError
        err = PearlsAQIError("simple error")
        assert str(err) == "simple error"


class TestLogger:
    """Verify that logging configuration works."""

    def test_logger_imports(self) -> None:
        from src.logger import logger
        assert logger is not None

    def test_logger_name(self) -> None:
        from src.logger import logger
        assert logger.name == "pearls_aqi"

    def test_logger_has_handlers(self) -> None:
        from src.logger import logger
        assert len(logger.handlers) >= 2  # console + file

    def test_logger_writes_to_file(self) -> None:
        from src.logger import logger
        logger.info("Phase 1 setup test — logging works")
        log_file = PROJECT_ROOT / "data" / "logs" / "pearls_aqi.log"
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "Phase 1 setup test" in content

    def test_setup_logger_idempotent(self) -> None:
        from src.logger import setup_logger
        logger1 = setup_logger("test_idempotent")
        handler_count = len(logger1.handlers)
        logger2 = setup_logger("test_idempotent")
        assert logger1 is logger2
        assert len(logger2.handlers) == handler_count
