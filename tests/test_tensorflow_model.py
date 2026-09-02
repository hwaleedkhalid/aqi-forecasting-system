"""Unit tests for TensorFlowAQIModel."""

import os

# Suppress TF noise in test output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import numpy as np
import pytest

from src.exceptions import ModelTrainingError
from src.models.tensorflow_model import TensorFlowAQIModel


class TestTensorFlowAQIModel:
    """Test TensorFlow DNN fitting, prediction, clamping, and serialization."""

    @pytest.fixture()
    def small_dataset(self):
        """Create a small deterministic dataset for fast unit tests."""
        np.random.seed(42)
        X = np.random.randn(60, 8).astype(np.float32)
        y = np.random.uniform(30, 200, (60, 12)).astype(np.float32)
        return X, y

    def test_fit_and_predict_shape(self, small_dataset) -> None:
        X, y = small_dataset
        model = TensorFlowAQIModel(
            hidden_layers=[
                {"units": 16, "activation": "relu", "dropout": 0.1},
                {"units": 8, "activation": "relu", "dropout": 0.0},
            ],
            epochs=3,
            batch_size=16,
            forecast_horizons=12,
            validation_fraction=0.2,
        )
        history = model.fit(X, y)

        assert model.is_fitted
        assert history.epochs_completed > 0

        preds = model.predict(X)
        assert preds.shape == (60, 12)

    def test_predictions_clamped_to_valid_range(self, small_dataset) -> None:
        X, y = small_dataset
        model = TensorFlowAQIModel(
            hidden_layers=[{"units": 8, "activation": "relu", "dropout": 0.0}],
            epochs=2,
            batch_size=32,
            forecast_horizons=12,
        )
        model.fit(X, y)
        preds = model.predict(X)

        assert np.all(preds >= 0.0)
        assert np.all(preds <= 500.0)

    def test_predict_unfitted_raises_error(self) -> None:
        model = TensorFlowAQIModel()
        with pytest.raises(ModelTrainingError, match="must be fitted"):
            model.predict(np.zeros((5, 10)))

    def test_explicit_validation_data(self, small_dataset) -> None:
        X, y = small_dataset
        X_val = np.random.randn(10, 8).astype(np.float32)
        y_val = np.random.uniform(30, 200, (10, 12)).astype(np.float32)

        model = TensorFlowAQIModel(
            hidden_layers=[{"units": 8, "activation": "relu", "dropout": 0.0}],
            epochs=2,
            batch_size=16,
            forecast_horizons=12,
        )
        history = model.fit(X, y, validation_data=(X_val, y_val))
        assert model.is_fitted
        assert history.best_val_loss < float("inf")

    def test_save_and_load(self, small_dataset, tmp_path: Path) -> None:
        X, y = small_dataset
        model = TensorFlowAQIModel(
            hidden_layers=[{"units": 8, "activation": "relu", "dropout": 0.0}],
            epochs=2,
            batch_size=32,
            forecast_horizons=12,
        )
        model.fit(X, y)

        model_file = tmp_path / "tf_model.keras"
        model.save(model_file)
        assert model_file.exists()

        loaded = TensorFlowAQIModel.load(model_file)
        assert loaded.is_fitted

        orig_preds = model.predict(X[:5])
        loaded_preds = loaded.predict(X[:5])
        np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-5)

    def test_model_name(self) -> None:
        model = TensorFlowAQIModel()
        assert model.name == "TensorFlow DNN"

    def test_get_model_summary_before_build(self) -> None:
        model = TensorFlowAQIModel()
        summary = model.get_model_summary()
        assert "not built" in summary.lower()
