"""Pearls AQI Predictor - TensorFlow Feed-Forward Neural Network Model.

Implements a multi-output Dense network to forecast 72 future hourly AQI
values, capturing non-linear global representations of pollutant dynamics.

Architecture (from config.TF_HIDDEN_LAYERS):
    Input(64) -> Dense(128, ReLU) + Dropout(0.3)
              -> Dense(64, ReLU) + Dropout(0.2)
              -> Dense(32, ReLU)
              -> Dense(72, Linear)

Training protocol:
    - Optimizer:  Adam with configurable learning rate.
    - Loss:       Mean Squared Error (consistent with RMSE-primary evaluation).
    - Callbacks:  EarlyStopping (restore_best_weights) + ReduceLROnPlateau.
    - Validation: Last chronological fraction of training data ONLY;
                  the held-out test partition is never seen during training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Suppress noisy TensorFlow/oneDNN warnings before import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf  # noqa: E402

from src.config import (
    AQI_MAX,
    AQI_MIN,
    FORECAST_HORIZONS,
    RANDOM_STATE,
    TF_BATCH_SIZE,
    TF_EARLY_STOPPING_PATIENCE,
    TF_EPOCHS,
    TF_HIDDEN_LAYERS,
    TF_LEARNING_RATE,
)
from src.exceptions import ModelTrainingError
from src.logger import logger
from src.models.base_model import BaseAQIModel


@dataclass
class TFTrainingHistory:
    """Stores training run metadata for reproducibility audits."""

    epochs_completed: int = 0
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    final_train_loss: float = float("inf")
    final_val_loss: float = float("inf")
    history: dict[str, list[float]] = field(default_factory=dict)


class TensorFlowAQIModel(BaseAQIModel):
    """Multi-output feed-forward neural network for 72-hour AQI forecasting."""

    def __init__(
        self,
        hidden_layers: list[dict] | None = None,
        epochs: int = TF_EPOCHS,
        batch_size: int = TF_BATCH_SIZE,
        learning_rate: float = TF_LEARNING_RATE,
        patience: int = TF_EARLY_STOPPING_PATIENCE,
        forecast_horizons: int = FORECAST_HORIZONS,
        random_state: int = RANDOM_STATE,
        validation_fraction: float = 0.15,
    ) -> None:
        """Initialize TensorFlowAQIModel.

        Args:
            hidden_layers: List of dicts with keys 'units', 'activation', 'dropout'.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size for gradient updates.
            learning_rate: Initial Adam learning rate.
            patience: Early stopping patience (epochs without val_loss improvement).
            forecast_horizons: Number of future hourly AQI targets (default: 72).
            random_state: Seed for weight initialization reproducibility.
            validation_fraction: Fraction of training data (chronological tail)
                reserved for early-stopping validation. The held-out test set
                is NEVER used during training.
        """
        self.hidden_layers = hidden_layers or TF_HIDDEN_LAYERS
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.forecast_horizons = forecast_horizons
        self.random_state = random_state
        self.validation_fraction = validation_fraction

        self.model: tf.keras.Model | None = None
        self.is_fitted: bool = False
        self.training_history: TFTrainingHistory = TFTrainingHistory()

    @property
    def name(self) -> str:
        """Human-readable model name."""
        return "TensorFlow DNN"

    def _build_model(self, n_features: int) -> tf.keras.Model:
        """Construct the Keras Sequential model architecture.

        Args:
            n_features: Number of input features.

        Returns:
            Compiled Keras model.
        """
        tf.keras.utils.set_random_seed(self.random_state)

        model = tf.keras.Sequential(name="AQI_DNN_72h")

        # Input layer
        model.add(tf.keras.layers.InputLayer(shape=(n_features,)))

        # Hidden layers from config
        for i, layer_cfg in enumerate(self.hidden_layers):
            model.add(
                tf.keras.layers.Dense(
                    units=layer_cfg["units"],
                    activation=layer_cfg["activation"],
                    kernel_initializer="he_normal",
                    name=f"dense_{i}",
                )
            )
            dropout_rate = layer_cfg.get("dropout", 0.0)
            if dropout_rate > 0:
                model.add(
                    tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")
                )

        # Output layer: 72 linear neurons (one per forecast horizon)
        model.add(
            tf.keras.layers.Dense(
                units=self.forecast_horizons,
                activation="linear",
                name="output",
            )
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> TFTrainingHistory:
        """Train the neural network with early stopping.

        The validation set for early stopping is carved from the chronological
        tail of the training data (last `validation_fraction`), ensuring the
        held-out test partition is never seen during training.

        If `validation_data` is explicitly provided, it overrides the automatic
        chronological split — useful for unit tests with small arrays.

        Args:
            X: Scaled feature matrix (n_samples, n_features).
            y: Target matrix (n_samples, forecast_horizons).
            validation_data: Optional explicit (X_val, y_val) tuple.

        Returns:
            TFTrainingHistory with training run metadata.
        """
        try:
            n_features = X.shape[1]
            self.model = self._build_model(n_features)

            # Chronological train/val split from training data only
            if validation_data is None:
                split_idx = int(len(X) * (1.0 - self.validation_fraction))
                X_fit, y_fit = X[:split_idx], y[:split_idx]
                X_val, y_val = X[split_idx:], y[split_idx:]
                logger.info(
                    f"TF train/val split: fit={X_fit.shape[0]}, "
                    f"val={X_val.shape[0]} ({self.validation_fraction:.0%} chronological tail)"
                )
            else:
                X_fit, y_fit = X, y
                X_val, y_val = validation_data

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.patience,
                    restore_best_weights=True,
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=max(3, self.patience // 3),
                    min_lr=1e-6,
                    verbose=1,
                ),
            ]

            history = self.model.fit(
                X_fit,
                y_fit,
                validation_data=(X_val, y_val),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=callbacks,
                verbose=1,
            )

            self.is_fitted = True

            # Record training metadata
            stopped_epoch = (
                callbacks[0].stopped_epoch if callbacks[0].stopped_epoch > 0
                else self.epochs
            )
            best_epoch = (
                stopped_epoch - self.patience
                if callbacks[0].stopped_epoch > 0
                else stopped_epoch
            )

            self.training_history = TFTrainingHistory(
                epochs_completed=stopped_epoch,
                best_epoch=max(1, best_epoch),
                best_val_loss=float(min(history.history["val_loss"])),
                final_train_loss=float(history.history["loss"][-1]),
                final_val_loss=float(history.history["val_loss"][-1]),
                history={k: [float(v) for v in vals] for k, vals in history.history.items()},
            )

            logger.info(
                f"TF training complete: {self.training_history.epochs_completed} epochs, "
                f"best_val_loss={self.training_history.best_val_loss:.4f} "
                f"at epoch {self.training_history.best_epoch}"
            )

            return self.training_history

        except Exception as err:
            raise ModelTrainingError(
                f"Failed fitting {self.name}: {err}", detail=str(err)
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 72 future hourly AQI values, clamped to [0, 500].

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Clamped prediction matrix (n_samples, forecast_horizons).
        """
        if not self.is_fitted or self.model is None:
            raise ModelTrainingError("Model must be fitted before calling predict()")

        raw_preds = self.model.predict(X, verbose=0)
        return np.clip(raw_preds, float(AQI_MIN), float(AQI_MAX))

    def save(self, path: Path) -> Path:
        """Save model to Keras native format.

        Args:
            path: Destination path (directory or .keras file).

        Returns:
            Path to saved artifact.
        """
        if self.model is None:
            raise ModelTrainingError("No model to save — call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info(f"Saved TensorFlow model to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "TensorFlowAQIModel":
        """Load a saved Keras model from disk.

        Args:
            path: Path to .keras model artifact.

        Returns:
            TensorFlowAQIModel instance with loaded weights.
        """
        path = Path(path)
        if not path.exists():
            raise ModelTrainingError(f"Model artifact not found at {path}")

        instance = cls()
        instance.model = tf.keras.models.load_model(path)
        instance.is_fitted = True
        logger.info(f"Loaded TensorFlow model from {path}")
        return instance

    def get_model_summary(self) -> str:
        """Return a string representation of the Keras model architecture."""
        if self.model is None:
            return "Model not built yet."
        lines: list[str] = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)
