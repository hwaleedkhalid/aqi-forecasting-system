"""Unit tests for Phase 10.5D forecasting architecture models."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.models.grouped_horizon_model import GroupedHorizonModel
from src.models.hybrid_specialist_model import (
    HybridAQISpecialistModel,
    PersistenceAwareHybridModel,
)
from src.models.pollutant_to_aqi_model import (
    CRITERIA_POLLUTANTS,
    MultiPollutantToAQIModel,
    calculate_vectorized_epa_aqi,
    calculate_vectorized_sub_index,
)


class TestVectorizedEPACalculator:
    """Test vectorized EPA sub-index and overall AQI computation."""

    def test_pm25_vectorized_sub_index(self) -> None:
        conc = np.array([[0.0, 9.0, 12.0, 35.4], [55.4, 125.4, 225.4, 400.0]])
        sub_indices = calculate_vectorized_sub_index("pm2_5", conc)
        assert sub_indices.shape == (2, 4)
        assert np.isclose(sub_indices[0, 0], 0.0)
        assert np.isclose(sub_indices[0, 1], 50.0)
        assert 50.0 < sub_indices[0, 2] < 100.0
        assert np.isclose(sub_indices[0, 3], 100.0)
        assert np.isclose(sub_indices[1, 0], 150.0)

    def test_overall_vectorized_aqi_max_operator(self) -> None:
        # Create dummy dict for 6 pollutants
        N, H = 5, 72
        preds = {pol: np.zeros((N, H)) for pol in CRITERIA_POLLUTANTS}
        # Set PM2.5 to 35.4 (AQI = 100)
        preds["pm2_5"][:] = 35.4
        # Set O3 in sample 0 to 166.0 (AQI = 150)
        preds["o3"][0, :] = 166.0

        overall_aqi = calculate_vectorized_epa_aqi(preds)
        assert overall_aqi.shape == (N, H)
        assert np.isclose(overall_aqi[0, 0], 150.0)  # O3 dominates
        assert np.isclose(overall_aqi[1, 0], 100.0)  # PM2.5 dominates


class TestMultiPollutantToAQIModel:
    """Test MultiPollutantToAQIModel training and prediction pipeline."""

    def test_fit_and_predict(self) -> None:
        N, D, H = 20, 10, 72
        X = np.random.randn(N, D).astype(np.float32)
        y_dict = {
            pol: np.random.uniform(5, 100, (N, H)).astype(np.float32)
            for pol in CRITERIA_POLLUTANTS
        }

        model = MultiPollutantToAQIModel()
        model.fit(X, y_dict)
        assert model.is_fitted

        preds = model.predict(X)
        assert preds.shape == (N, H)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 500.0)


class TestGroupedHorizonModel:
    """Test GroupedHorizonModel."""

    def test_fit_and_predict(self) -> None:
        N, D, H = 20, 10, 72
        X = np.random.randn(N, D).astype(np.float32)
        y = np.random.uniform(50, 200, (N, H)).astype(np.float32)

        model = GroupedHorizonModel()
        model.fit(X, y)
        assert model.is_fitted

        preds = model.predict(X)
        assert preds.shape == (N, H)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 500.0)


class TestHybridSpecialistModels:
    """Test HybridAQISpecialistModel and PersistenceAwareHybridModel."""

    def test_hybrid_specialist(self) -> None:
        N, D, H = 20, 10, 72
        X = np.random.randn(N, D).astype(np.float32)
        y = np.random.uniform(50, 200, (N, H)).astype(np.float32)

        model = HybridAQISpecialistModel(short_estimators=5)
        model.fit(X, y)
        assert model.is_fitted

        preds = model.predict(X)
        assert preds.shape == (N, H)

    def test_persistence_aware_hybrid(self) -> None:
        N, D, H = 20, 10, 72
        X = np.random.randn(N, D).astype(np.float32)
        y = np.random.uniform(50, 200, (N, H)).astype(np.float32)
        curr_aqi = np.random.uniform(50, 200, N).astype(np.float32)

        model = PersistenceAwareHybridModel(short_estimators=5, min_blend_weight=0.5)
        model.fit(X, y)
        assert model.is_fitted

        preds = model.predict(X, current_aqi=curr_aqi)
        assert preds.shape == (N, H)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 500.0)
