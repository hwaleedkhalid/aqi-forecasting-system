"""Pearls AQI Predictor - Model Trainer.

Orchestrates model training, baseline evaluation, comparison tracking, and model artifact serialization.
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.exceptions import ModelTrainingError, ValidationError
from src.logger import logger
from src.models.naive_baseline import NaivePersistenceBaseline
from src.models.ridge_model import RidgeAQIModel
from src.training_pipeline.evaluator import ModelEvaluator


class ModelTrainer:
    """Orchestrates model training pipelines and comparative evaluation."""

    def __init__(
        self,
        data_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        """Initialize ModelTrainer.

        Args:
            data_dir: Directory containing preprocessed .npy arrays.
            models_dir: Directory for storing trained model artifacts.
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.evaluations: dict[str, dict[str, Any]] = {}

    def load_datasets(self) -> dict[str, np.ndarray]:
        """Load preprocessed NumPy arrays for training and evaluation.

        Returns:
            Dictionary containing X_train, y_train, X_test, y_test, current_aqi_train, current_aqi_test.
        """
        req_files = [
            "X_train.npy",
            "y_train.npy",
            "X_test.npy",
            "y_test.npy",
            "current_aqi_train.npy",
            "current_aqi_test.npy",
        ]
        missing = [f for f in req_files if not (self.data_dir / f).exists()]
        if missing:
            raise ValidationError(
                f"Missing required dataset files in {self.data_dir}: {missing}",
                detail="Run DatasetBuilder before training models.",
            )

        data = {
            "X_train": np.load(self.data_dir / "X_train.npy"),
            "y_train": np.load(self.data_dir / "y_train.npy"),
            "X_test": np.load(self.data_dir / "X_test.npy"),
            "y_test": np.load(self.data_dir / "y_test.npy"),
            "current_aqi_train": np.load(self.data_dir / "current_aqi_train.npy"),
            "current_aqi_test": np.load(self.data_dir / "current_aqi_test.npy"),
        }
        logger.info(
            f"Loaded datasets: X_train={data['X_train'].shape}, y_train={data['y_train'].shape}, "
            f"X_test={data['X_test'].shape}, y_test={data['y_test'].shape}"
        )
        return data

    def run_ridge_pipeline(self) -> tuple[RidgeAQIModel, pd.DataFrame]:
        """Execute Phase 8: Naive Baseline and Ridge Regression training pipeline.

        Returns:
            Tuple of (trained_ridge_model, comparison_dataframe).
        """
        data = self.load_datasets()
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        current_test = data["current_aqi_test"]

        # 1. Evaluate Naive Persistence Baseline
        logger.info("Evaluating Naive Persistence Baseline on test set...")
        baseline = NaivePersistenceBaseline(forecast_horizons=y_test.shape[1])
        y_pred_baseline = baseline.predict_from_current(current_test)
        self.evaluations[baseline.name] = ModelEvaluator.evaluate_predictions(
            y_test, y_pred_baseline, model_name=baseline.name
        )

        # 2. Train and Evaluate Ridge Regression
        logger.info("Training Ridge Regression (MultiOutputRegressor)...")
        ridge_model = RidgeAQIModel()
        ridge_model.fit(X_train, y_train)

        y_pred_ridge = ridge_model.predict(X_test)
        self.evaluations[ridge_model.name] = ModelEvaluator.evaluate_predictions(
            y_test, y_pred_ridge, model_name=ridge_model.name
        )

        # 3. Save Ridge Model Artifact
        model_path = self.models_dir / "ridge_model.joblib"
        ridge_model.save(model_path)
        logger.info(f"Saved trained Ridge model to {model_path}")

        # 4. Generate Comparison Report
        df_comparison = ModelEvaluator.compare_models(self.evaluations)
        ModelEvaluator.save_comparison(
            self.evaluations, self.models_dir / "model_comparison.json"
        )

        logger.info("\n=== Model Comparison on Test Partition ===\n" + df_comparison.to_string(index=False))

        return ridge_model, df_comparison
