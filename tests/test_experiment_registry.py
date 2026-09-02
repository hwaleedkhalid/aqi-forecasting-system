"""Unit tests for ExperimentRegistry, ChronologicalCVEvaluator, and LightGBM models."""

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.models.lightgbm_models import (
    LightGBMDirectMultiOutput,
    LightGBMHorizonAsFeature,
)
from src.training_pipeline.cv_evaluator import ChronologicalCVEvaluator
from src.training_pipeline.experiment_registry import (
    ExperimentRecord,
    ExperimentRegistry,
)


class TestExperimentRegistry:
    """Test experiment logging, persistence, leaderboard, and retrieval."""

    def test_log_and_retrieve_best_experiment(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(registry_dir=tmp_path)

        rec1 = ExperimentRecord(
            experiment_id="EXP-001",
            feature_version="v2_weather_enriched",
            model_name="Ridge",
            hyperparameters={"alpha": 1.0},
            strategy="direct",
            cv_scheme="3fold",
            val_rmse_mean=92.5,
            val_rmse_std=1.2,
            val_mae_mean=69.0,
            val_r2_mean=0.51,
            val_h1_rmse=53.0,
            val_h24_rmse=77.0,
            val_h72_rmse=99.0,
            training_time_sec=0.5,
        )
        rec2 = ExperimentRecord(
            experiment_id="EXP-002",
            feature_version="v2_weather_enriched",
            model_name="Ridge",
            hyperparameters={"alpha": 10.0},
            strategy="direct",
            cv_scheme="3fold",
            val_rmse_mean=89.3,
            val_rmse_std=0.8,
            val_mae_mean=66.5,
            val_r2_mean=0.55,
            val_h1_rmse=51.2,
            val_h24_rmse=74.1,
            val_h72_rmse=96.0,
            training_time_sec=0.4,
        )

        registry.log_experiment(rec1)
        registry.log_experiment(rec2)

        df_board = registry.get_leaderboard()
        assert len(df_board) == 2
        assert df_board.iloc[0]["Experiment ID"] == "EXP-002"  # lower RMSE first

        best = registry.get_best_experiment()
        assert best is not None
        assert best.experiment_id == "EXP-002"


class TestChronologicalCVEvaluator:
    """Test fold generation and evaluation mechanics."""

    def test_get_folds_expanding_indices(self) -> None:
        evaluator = ChronologicalCVEvaluator(n_folds=3, min_train_fraction=0.50, val_fraction_per_fold=0.15)
        folds = evaluator.get_folds(100)

        assert len(folds) == 3
        # Verify fold 1
        train1, val1 = folds[0]
        assert len(train1) == 50
        assert val1[0] == 50
        assert val1[-1] == 64

    def test_evaluate_model_cv_ridge(self) -> None:
        X = np.random.randn(80, 10).astype(np.float32)
        y = np.random.uniform(50, 200, (80, 72)).astype(np.float32)

        evaluator = ChronologicalCVEvaluator(n_folds=2, min_train_fraction=0.6, val_fraction_per_fold=0.2)
        res = evaluator.evaluate_model_cv(lambda: Ridge(alpha=1.0), X, y)

        assert "val_rmse_mean" in res
        assert "val_mae_mean" in res
        assert res["val_rmse_mean"] > 0


class TestLightGBMModels:
    """Test LightGBM direct and horizon-as-feature implementations."""

    @pytest.fixture
    def small_data(self):
        np.random.seed(42)
        X = np.random.randn(30, 8).astype(np.float32)
        y = np.random.uniform(30, 250, (30, 72)).astype(np.float32)
        return X, y

    def test_lightgbm_direct_fit_and_predict(self, small_data) -> None:
        X, y = small_data
        model = LightGBMDirectMultiOutput(n_estimators=5, num_leaves=7)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30, 72)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 500.0)

    def test_lightgbm_horizon_as_feature_fit_and_predict(self, small_data) -> None:
        X, y = small_data
        model = LightGBMHorizonAsFeature(n_estimators=5, num_leaves=7)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (30, 72)
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 500.0)
