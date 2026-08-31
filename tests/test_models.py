"""Unit tests for Forecasting Models (BaseModel, NaivePersistenceBaseline, RidgeAQIModel)."""

from pathlib import Path
import numpy as np
import pytest

from src.exceptions import ModelTrainingError, ValidationError
from src.models.base_model import BaseAQIModel
from src.models.naive_baseline import NaivePersistenceBaseline
from src.models.ridge_model import RidgeAQIModel


# =============================================================================
# Naive Persistence Baseline Tests
# =============================================================================

class TestNaivePersistenceBaseline:
    """Test Naive Persistence Baseline logic and shapes."""

    def test_predict_from_current_broadcasts_72_horizons(self) -> None:
        baseline = NaivePersistenceBaseline(forecast_horizons=72)
        current_aqi = np.array([50.0, 120.0, 300.0])  # 3 samples
        preds = baseline.predict_from_current(current_aqi)

        assert preds.shape == (3, 72)
        # All columns for sample 0 must be 50.0
        assert np.all(preds[0, :] == 50.0)
        assert np.all(preds[1, :] == 120.0)
        assert np.all(preds[2, :] == 300.0)

    def test_predict_method_accepts_1d_and_2d_single_column(self) -> None:
        baseline = NaivePersistenceBaseline(forecast_horizons=24)
        # 1D array
        p1 = baseline.predict(np.array([45.0, 80.0]))
        assert p1.shape == (2, 24)

        # 2D (N, 1) array
        p2 = baseline.predict(np.array([[45.0], [80.0]]))
        assert p2.shape == (2, 24)

    def test_predict_invalid_shape_raises_validation_error(self) -> None:
        baseline = NaivePersistenceBaseline(forecast_horizons=24)
        with pytest.raises(ValidationError, match="predict expects 1D array"):
            baseline.predict(np.ones((5, 10)))

    def test_baseline_save_and_load(self, tmp_path: Path) -> None:
        baseline = NaivePersistenceBaseline(forecast_horizons=72)
        save_file = tmp_path / "baseline_meta.json"
        baseline.save(save_file)

        assert save_file.exists()
        loaded = NaivePersistenceBaseline().load(save_file)
        assert loaded.forecast_horizons == 72


# =============================================================================
# Ridge AQI Model Tests
# =============================================================================

class TestRidgeAQIModel:
    """Test RidgeAQIModel fitting, multi-output prediction, clamping, and serialization."""

    def test_fit_and_predict_multi_output(self) -> None:
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.uniform(20, 300, (50, 72))

        model = RidgeAQIModel(alpha=1.0)
        model.fit(X, y)

        assert model.is_fitted
        preds = model.predict(X)
        assert preds.shape == (50, 72)
        # Ensure values are clamped in [0, 500]
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 500.0)

    def test_predict_unfitted_model_raises_error(self) -> None:
        model = RidgeAQIModel()
        with pytest.raises(ModelTrainingError, match="must be fitted before calling predict"):
            model.predict(np.zeros((5, 10)))

    def test_get_coefficients(self) -> None:
        X = np.random.randn(30, 4)
        y = np.random.uniform(50, 150, (30, 72))
        features = ["f1", "f2", "f3", "f4"]

        model = RidgeAQIModel()
        model.fit(X, y)
        coefs = model.get_coefficients(features)

        assert len(coefs) == 4
        assert len(coefs["f1"]) == 72

    def test_save_and_load_ridge_model(self, tmp_path: Path) -> None:
        X = np.random.randn(20, 5)
        y = np.random.uniform(50, 150, (20, 12))

        model = RidgeAQIModel()
        model.fit(X, y)

        model_file = tmp_path / "ridge_model.joblib"
        model.save(model_file)

        assert model_file.exists()
        loaded = RidgeAQIModel.load(model_file)
        assert loaded.is_fitted
        preds = loaded.predict(X)
        assert preds.shape == (20, 12)
