"""Unit tests for Model Evaluator and Trainer."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.exceptions import ValidationError
from src.training_pipeline.evaluator import ModelEvaluator
from src.training_pipeline.trainer import ModelTrainer


class TestModelEvaluator:
    """Test regression metric computation across all prediction horizons."""

    def test_evaluate_predictions_metrics(self) -> None:
        y_true = np.array([
            [100.0, 110.0, 120.0],
            [200.0, 210.0, 220.0],
        ])
        y_pred = np.array([
            [105.0, 115.0, 125.0],
            [195.0, 205.0, 215.0],
        ])

        eval_dict = ModelEvaluator.evaluate_predictions(y_true, y_pred, model_name="TestModel")

        assert eval_dict["model_name"] == "TestModel"
        assert np.isclose(eval_dict["overall_mae"], 5.0)
        assert np.isclose(eval_dict["overall_rmse"], 5.0)
        assert "h1" in eval_dict["all_horizons"]
        assert "h3" in eval_dict["all_horizons"]

    def test_compare_models_summary(self, tmp_path: Path) -> None:
        y_true = np.ones((10, 6)) * 100
        y_pred_baseline = np.ones((10, 6)) * 120  # Error = 20
        y_pred_model = np.ones((10, 6)) * 110     # Error = 10 (50% improvement)

        eval_baseline = ModelEvaluator.evaluate_predictions(
            y_true, y_pred_baseline, "Naive Persistence Baseline"
        )
        eval_model = ModelEvaluator.evaluate_predictions(
            y_true, y_pred_model, "Ridge Model"
        )

        evaluations = {
            "Naive Persistence Baseline": eval_baseline,
            "Ridge Model": eval_model,
        }

        df_comp = ModelEvaluator.compare_models(evaluations)
        assert len(df_comp) == 2
        assert df_comp.loc[df_comp["Model"] == "Ridge Model", "vs Baseline (% RMSE)"].iloc[0] == "+50.00%"

        save_path = tmp_path / "comp.json"
        ModelEvaluator.save_comparison(evaluations, save_path)
        assert save_path.exists()


class TestModelTrainer:
    """Test ModelTrainer data loading and execution."""

    def test_missing_data_files_raises_validation_error(self, tmp_path: Path) -> None:
        trainer = ModelTrainer(data_dir=tmp_path, models_dir=tmp_path / "models")
        with pytest.raises(ValidationError, match="Missing required dataset files"):
            trainer.load_datasets()

    def test_run_ridge_pipeline(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        models_dir = tmp_path / "models"
        data_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        np.random.seed(42)
        X_train = np.random.randn(40, 5)
        y_train = np.random.uniform(50, 150, (40, 6))
        X_test = np.random.randn(10, 5)
        y_test = np.random.uniform(50, 150, (10, 6))
        current_train = np.random.uniform(50, 150, 40)
        current_test = np.random.uniform(50, 150, 10)

        np.save(data_dir / "X_train.npy", X_train)
        np.save(data_dir / "y_train.npy", y_train)
        np.save(data_dir / "X_test.npy", X_test)
        np.save(data_dir / "y_test.npy", y_test)
        np.save(data_dir / "current_aqi_train.npy", current_train)
        np.save(data_dir / "current_aqi_test.npy", current_test)

        trainer = ModelTrainer(data_dir=data_dir, models_dir=models_dir)
        ridge_model, df_comp = trainer.run_ridge_pipeline()

        assert ridge_model.is_fitted
        assert (models_dir / "ridge_model.joblib").exists()
        assert (models_dir / "model_comparison.json").exists()
        assert len(df_comp) == 2
