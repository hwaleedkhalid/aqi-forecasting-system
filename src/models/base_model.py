"""Pearls AQI Predictor - Abstract Base Model.

Defines the contract for multi-output forecasting models (Naive Baseline,
Ridge Regression, Random Forest, TensorFlow Neural Network).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np


class BaseAQIModel(ABC):
    """Abstract base class for AQI multi-output forecasting models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model on training features and targets.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Target matrix of shape (n_samples, forecast_horizons).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate multi-output forecast predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted AQI matrix of shape (n_samples, forecast_horizons).
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> Path:
        """Serialize model artifact to disk.

        Args:
            path: Destination file path.

        Returns:
            Path to saved artifact.
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> "BaseAQIModel":
        """Load model artifact from disk.

        Args:
            path: Artifact file path.

        Returns:
            Loaded model instance.
        """
        pass
