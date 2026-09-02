"""Model package — base definitions and model implementations."""

from src.models.base_model import BaseAQIModel
from src.models.lightgbm_models import (
    LightGBMDirectMultiOutput,
    LightGBMHorizonAsFeature,
)
from src.models.naive_baseline import NaivePersistenceBaseline
from src.models.random_forest_model import RandomForestAQIModel
from src.models.ridge_model import RidgeAQIModel


def get_tensorflow_model_class():
    """Lazy-load TensorFlowAQIModel to avoid heavy TensorFlow initialization overhead."""
    from src.models.tensorflow_model import TensorFlowAQIModel
    return TensorFlowAQIModel


def __getattr__(name: str):
    if name == "TensorFlowAQIModel":
        return get_tensorflow_model_class()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseAQIModel",
    "NaivePersistenceBaseline",
    "RidgeAQIModel",
    "RandomForestAQIModel",
    "LightGBMDirectMultiOutput",
    "LightGBMHorizonAsFeature",
    "TensorFlowAQIModel",
    "get_tensorflow_model_class",
]
