"""Pearls AQI Predictor - Custom Exception Classes.

Domain-specific exceptions for clean error handling across the application.
Defined per Rules.md §4: Error Handling Strategy.
"""


class PearlsAQIError(Exception):
    """Base exception for all Pearls AQI Predictor errors."""

    def __init__(self, message: str = "", detail: str = "") -> None:
        self.message = message
        self.detail = detail
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} | Detail: {self.detail}"
        return self.message


class DataIngestionError(PearlsAQIError):
    """Failed to fetch data from an external API.

    Raised when an API call fails after all retry attempts,
    or when the API response is malformed or unexpected.
    """
    pass


class FeatureStoreError(PearlsAQIError):
    """Failed to read from or write to Hopsworks Feature Store.

    Raised during feature group creation, feature insertion,
    feature view queries, or any Hopsworks SDK operation.
    """
    pass


class ModelTrainingError(PearlsAQIError):
    """Failed during model training.

    Raised when model fitting fails, when training data is invalid,
    or when model serialization/deserialization encounters errors.
    """
    pass


class PredictionError(PearlsAQIError):
    """Failed to generate predictions.

    Raised when the inference pipeline cannot produce forecasts,
    e.g., model not loaded, features unavailable, or prediction
    values are out of valid range.
    """
    pass


class ValidationError(PearlsAQIError):
    """Data validation failed.

    Raised when incoming data (API responses, feature values, or
    user inputs) fails validation checks such as null values,
    out-of-range values, or missing required fields.
    """
    pass
