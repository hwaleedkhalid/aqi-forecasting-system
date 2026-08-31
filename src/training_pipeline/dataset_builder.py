"""Pearls AQI Predictor - Multi-Output Training Dataset Builder.

Constructs aligned 72-hour multi-output target matrices [y_{t+1}, ..., y_{t+72}],
enforces strict chronological train/test splitting with an anti-leakage embargo gap,
fits feature scalers exclusively on training data, and serializes dataset artifacts.
"""

import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    FORECAST_HORIZONS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_TEST_SPLIT_RATIO,
)
from src.exceptions import ValidationError
from src.logger import logger


class DatasetBuilder:
    """Builds and serializes multi-output training and evaluation datasets."""

    def __init__(
        self,
        forecast_horizons: int = FORECAST_HORIZONS,
        train_ratio: float = TRAIN_TEST_SPLIT_RATIO,
        target_col: str = "epa_aqi",
        output_dir: Path = PROCESSED_DATA_DIR,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        """Initialize DatasetBuilder.

        Args:
            forecast_horizons: Number of future hourly forecast steps (default: 72).
            train_ratio: Chronological train split fraction (default: 0.80).
            target_col: Target variable column name (default: 'epa_aqi').
            output_dir: Directory for storing processed dataset artifacts.
            models_dir: Directory for storing fitted scaler artifacts.
        """
        self.forecast_horizons = forecast_horizons
        self.train_ratio = train_ratio
        self.target_col = target_col
        self.output_dir = output_dir
        self.models_dir = models_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def construct_multi_output_targets(
        self, df: pd.DataFrame, feature_names: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
        """Construct multi-step future targets Y = [y_{t+1}, ..., y_{t+H}].

        Validates that all H forecast steps are continuous and non-null. Rows
        where the future horizon crosses missing data gaps or the end of the
        series are cleanly dropped.

        Args:
            df: Continuous hourly DataFrame with DatetimeIndex containing target_col.
            feature_names: Ordered list of feature column names.

        Returns:
            Tuple of (X_features_df, Y_targets_df, valid_sample_timestamps).

        Raises:
            ValidationError: If target_col is missing from input DataFrame.
        """
        if self.target_col not in df.columns:
            raise ValidationError(
                f"Target column '{self.target_col}' not found in DataFrame",
                detail=f"Available columns: {list(df.columns[:10])}...",
            )

        logger.info(
            f"Constructing {self.forecast_horizons}-hour multi-output targets for '{self.target_col}'..."
        )

        target_cols = [f"target_h{h}" for h in range(1, self.forecast_horizons + 1)]
        target_dict = {}

        # Shift target backwards in time: shift(-h) aligns observation at t+h with row t
        for h in range(1, self.forecast_horizons + 1):
            col_name = f"target_h{h}"
            target_dict[col_name] = df[self.target_col].shift(-h)

        df_targets = pd.DataFrame(target_dict, index=df.index)

        # Drop samples where future target horizon has missing data or ends
        valid_mask = ~df_targets.isna().any(axis=1) & ~df[feature_names].isna().any(axis=1)

        X_valid = df.loc[valid_mask, feature_names].copy()
        Y_valid = df_targets.loc[valid_mask, target_cols].copy()
        valid_timestamps = df.index[valid_mask]

        dropped_count = len(df) - len(X_valid)
        logger.info(
            f"Total samples with complete {self.forecast_horizons}h targets: {len(X_valid):,} "
            f"(Dropped {dropped_count:,} boundary/gap rows)"
        )

        return X_valid, Y_valid, valid_timestamps

    def split_train_test_chronological(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        embargo_hours: int | None = None,
    ) -> dict[str, Any]:
        """Perform chronological train/test split with an anti-leakage embargo gap.

        An embargo gap equal to forecast_horizons (72h) is enforced before the
        test boundary to ensure that no training target window extends into the
        test evaluation period.

        Args:
            X: Aligned feature DataFrame.
            Y: Aligned multi-output target DataFrame.
            timestamps: Sample timestamps.
            embargo_hours: Gap hours between train and test. Defaults to forecast_horizons.

        Returns:
            Dictionary containing train/test dataframes, timestamps, and split metadata.
        """
        if embargo_hours is None:
            embargo_hours = self.forecast_horizons

        total_samples = len(timestamps)
        split_idx = int(total_samples * self.train_ratio)
        split_timestamp = timestamps[split_idx]

        # Embargo cutoff: last allowed training sample timestamp
        train_cutoff = split_timestamp - pd.Timedelta(hours=embargo_hours)

        train_mask = timestamps <= train_cutoff
        test_mask = timestamps >= split_timestamp

        X_train_raw = X.loc[train_mask].copy()
        Y_train = Y.loc[train_mask].copy()
        train_times = timestamps[train_mask]

        X_test_raw = X.loc[test_mask].copy()
        Y_test = Y.loc[test_mask].copy()
        test_times = timestamps[test_mask]

        embargoed_count = total_samples - (len(X_train_raw) + len(X_test_raw))

        logger.info(
            f"Chronological Split: {self.train_ratio*100:.0f}% Train / {(1-self.train_ratio)*100:.0f}% Test"
        )
        logger.info(f"Split Boundary Timestamp: {split_timestamp} UTC")
        logger.info(f"Train Period: {train_times.min()} -> {train_times.max()} ({len(X_train_raw):,} samples)")
        logger.info(f"Embargo Gap: {embargo_hours} hours ({embargoed_count} samples purged to prevent leakage)")
        logger.info(f"Test Period: {test_times.min()} -> {test_times.max()} ({len(X_test_raw):,} samples)")

        return {
            "X_train_raw": X_train_raw,
            "Y_train": Y_train,
            "train_timestamps": train_times,
            "X_test_raw": X_test_raw,
            "Y_test": Y_test,
            "test_timestamps": test_times,
            "split_timestamp": split_timestamp.isoformat(),
            "embargo_hours": embargo_hours,
            "embargoed_samples_count": embargoed_count,
        }

    def fit_and_apply_scaler(
        self,
        X_train_raw: pd.DataFrame,
        X_test_raw: pd.DataFrame,
        feature_names: list[str],
        save_scaler: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Fit StandardScaler exclusively on training features and scale datasets.

        Args:
            X_train_raw: Raw training features.
            X_test_raw: Raw test features.
            feature_names: List of feature column names.
            save_scaler: If True, saves fitted scaler to data/models/feature_scaler.joblib.

        Returns:
            Tuple of (X_train_scaled, X_test_scaled, fitted_scaler).
        """
        scaler = StandardScaler()
        # FIT ONLY ON TRAINING SET
        X_train_scaled = scaler.fit_transform(X_train_raw[feature_names].values)
        # TRANSFORM TEST SET WITHOUT FITTING
        X_test_scaled = scaler.transform(X_test_raw[feature_names].values)

        if save_scaler:
            scaler_path = self.models_dir / "feature_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved fitted feature scaler to {scaler_path}")

        return X_train_scaled, X_test_scaled, scaler

    def build_and_save_dataset(
        self,
        df_features: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, Any]:
        """Execute full dataset construction, splitting, scaling, and artifact saving.

        Args:
            df_features: Processed DataFrame containing features and target.
            feature_names: Ordered list of feature names.

        Returns:
            Summary dictionary of the prepared dataset.
        """
        logger.info("Building multi-output supervised training dataset...")

        X, Y, timestamps = self.construct_multi_output_targets(df_features, feature_names)

        split_data = self.split_train_test_chronological(X, Y, timestamps)

        X_train_scaled, X_test_scaled, scaler = self.fit_and_apply_scaler(
            split_data["X_train_raw"],
            split_data["X_test_raw"],
            feature_names,
            save_scaler=True,
        )

        # Extract unscaled ground truth current target value y_t for naive baseline
        current_y = df_features.loc[timestamps, self.target_col]
        current_aqi_train = current_y.loc[split_data["train_timestamps"]].values
        current_aqi_test = current_y.loc[split_data["test_timestamps"]].values

        Y_train_arr = split_data["Y_train"].values
        Y_test_arr = split_data["Y_test"].values

        # Persist NumPy binary arrays
        np.save(self.output_dir / "X_train.npy", X_train_scaled)
        np.save(self.output_dir / "y_train.npy", Y_train_arr)
        np.save(self.output_dir / "X_test.npy", X_test_scaled)
        np.save(self.output_dir / "y_test.npy", Y_test_arr)
        np.save(self.output_dir / "current_aqi_train.npy", current_aqi_train)
        np.save(self.output_dir / "current_aqi_test.npy", current_aqi_test)

        # Persist raw tabular DataFrames (CSV)
        split_data["X_train_raw"].to_csv(self.output_dir / "X_train_raw.csv", index=True)
        split_data["X_test_raw"].to_csv(self.output_dir / "X_test_raw.csv", index=True)
        split_data["Y_train"].to_csv(self.output_dir / "y_train.csv", index=True)
        split_data["Y_test"].to_csv(self.output_dir / "y_test.csv", index=True)

        # Persist timestamps and ground-truth current AQI for evaluation/baseline reproducibility
        pd.DataFrame({
            "datetime_utc": split_data["train_timestamps"],
            "current_aqi": current_aqi_train,
        }).to_csv(self.output_dir / "train_timestamps.csv", index=False)

        pd.DataFrame({
            "datetime_utc": split_data["test_timestamps"],
            "current_aqi": current_aqi_test,
        }).to_csv(self.output_dir / "test_timestamps.csv", index=False)

        total_samples = len(timestamps)
        summary = {
            "target_variable": self.target_col,
            "forecast_horizons": self.forecast_horizons,
            "total_features": len(feature_names),
            "feature_names": feature_names,
            "total_supervised_pairs": total_samples,
            "nominal_split_ratio": {
                "train": self.train_ratio,
                "test": round(1 - self.train_ratio, 2),
            },
            "actual_retained_counts": {
                "train_samples": len(X_train_scaled),
                "test_samples": len(X_test_scaled),
                "embargoed_samples": split_data["embargoed_samples_count"],
            },
            "actual_retained_percentages": {
                "train_pct": round(len(X_train_scaled) / total_samples * 100, 2),
                "test_pct": round(len(X_test_scaled) / total_samples * 100, 2),
                "embargo_pct": round(split_data["embargoed_samples_count"] / total_samples * 100, 2),
            },
            "X_train_shape": list(X_train_scaled.shape),
            "y_train_shape": list(Y_train_arr.shape),
            "X_test_shape": list(X_test_scaled.shape),
            "y_test_shape": list(Y_test_arr.shape),
            "train_period": {
                "start_utc": str(split_data["train_timestamps"].min()),
                "end_utc": str(split_data["train_timestamps"].max()),
            },
            "test_period": {
                "start_utc": str(split_data["test_timestamps"].min()),
                "end_utc": str(split_data["test_timestamps"].max()),
            },
            "split_timestamp_utc": split_data["split_timestamp"],
            "embargo_gap_hours": split_data["embargo_hours"],
        }

        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved dataset summary to {summary_file}")
        logger.info(
            f"Dataset Ready: X_train={X_train_scaled.shape}, y_train={Y_train_arr.shape}, "
            f"X_test={X_test_scaled.shape}, y_test={Y_test_arr.shape}"
        )

        return summary
