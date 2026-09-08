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
from src.exceptions import FeatureStoreError, ValidationError
from src.feature_pipeline.hopsworks_integration import (
    HopsworksFeatureStoreConnector,
    load_canonical_feature_names,
)
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


class FeatureStoreTrainingLoader:
    """Retrieves, validates, and prepares historical training data from Hopsworks Feature Store.

    Enforces:
    - Bounded query timeouts on Arrow Flight offline reads.
    - Auditable duplicate detection (identical duplicates deduplicated; conflicting duplicates fail validation).
    - Preservation of protected holdout (candidate training/validation restricted strictly < 2025-06-07T00:00:00Z).
    - Exact physical timestamp semantics for multi-horizon targets (T + h*3600; drop if missing).
    - Strict chronological train/validation splitting with anti-leakage embargo gap.
    """

    HOLDOUT_START_DT: int = 1749254400  # 2025-06-07T00:00:00+00:00 UTC

    def __init__(
        self,
        connector: HopsworksFeatureStoreConnector | None = None,
        location_id: str = "lahore",
        forecast_horizons: int = FORECAST_HORIZONS,
        read_timeout: int = 300,
    ) -> None:
        """Initialize FeatureStoreTrainingLoader.

        Args:
            connector: Hopsworks connector instance (lazily initialized if None).
            location_id: Location primary key (default: 'lahore').
            forecast_horizons: Number of forecasting steps (default: 72).
            read_timeout: Bounded timeout in seconds for Arrow Flight query (default: 300).
        """
        self.connector = connector or HopsworksFeatureStoreConnector(location_id=location_id)
        self.location_id = location_id
        self.forecast_horizons = forecast_horizons
        self.read_timeout = read_timeout
        self.canonical_features = load_canonical_feature_names()

    def fetch_offline_data(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch full historical feature group from Hopsworks offline store with bounded timeout.

        Returns:
            Tuple of (DataFrame from offline store, query metadata dictionary).

        Raises:
            FeatureStoreError: If credentials fail, query times out, or dataset is empty.
        """
        if not self.connector.is_cloud_configured():
            raise FeatureStoreError("Hopsworks credentials not configured for offline training retrieval.")

        logger.info(
            f"Querying Hopsworks offline Feature Store for location_id='{self.location_id}' "
            f"(feature_group='{self.connector.feature_group_name}', version={self.connector.feature_group_version}, "
            f"bounded_timeout={self.read_timeout}s)..."
        )

        try:
            fs = self.connector._login()
            fg = fs.get_feature_group(
                name=self.connector.feature_group_name,
                version=self.connector.feature_group_version,
            )
            query = fg.select_all().filter(fg.location_id == self.location_id)
            read_opts = {"timeout": self.read_timeout}
            raw_df = query.read(read_options=read_opts)
        except Exception as e:
            logger.error(f"Hopsworks offline Feature Store read failed: {e}")
            raise FeatureStoreError(f"Hopsworks offline read failed or timed out: {e}") from e

        if raw_df is None or len(raw_df) == 0:
            raise FeatureStoreError("Hopsworks query returned 0 records.")

        # Inspect latest materialized observation in offline store
        latest_mat_dt = int(raw_df["dt"].max())
        earliest_mat_dt = int(raw_df["dt"].min())
        now_epoch = int(pd.Timestamp.now(tz="UTC").timestamp())
        mat_age_hours = round((now_epoch - latest_mat_dt) / 3600.0, 2)

        meta = {
            "source": "Hopsworks Feature Store (Offline Store)",
            "feature_group": self.connector.feature_group_name,
            "feature_group_version": self.connector.feature_group_version,
            "location_id": self.location_id,
            "raw_row_count": len(raw_df),
            "earliest_materialized_dt": earliest_mat_dt,
            "latest_materialized_dt": latest_mat_dt,
            "materialized_age_hours": mat_age_hours,
            "earliest_materialized_utc": pd.to_datetime(earliest_mat_dt, unit="s", utc=True).isoformat(),
            "latest_materialized_utc": pd.to_datetime(latest_mat_dt, unit="s", utc=True).isoformat(),
        }
        logger.info(
            f"Successfully retrieved {len(raw_df)} records from offline Feature Store. "
            f"Latest materialized dt={latest_mat_dt} ({mat_age_hours}h ago)."
        )
        return raw_df, meta

    def audit_and_clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Audit schema, validate numerical finiteness, and handle duplicates auditably.

        Args:
            df: Raw DataFrame retrieved from Feature Store or offline storage.

        Returns:
            Tuple of (cleaned DataFrame sorted by dt, audit report dictionary).

        Raises:
            ValidationError: If schema columns are missing, timestamps non-integral,
                            conflicting duplicates exist, or NaN/Inf values present.
        """
        # 1. Verify location_id and all canonical features exist
        missing = [c for c in self.canonical_features if c not in df.columns]
        if missing:
            raise ValidationError(
                f"Feature Store schema error: missing canonical columns: {missing[:5]}"
            )

        # 2. Validate dt column
        dt_numeric = pd.to_numeric(df["dt"], errors="coerce")
        if dt_numeric.isna().any():
            raise ValidationError("Non-numeric or NaN values detected in event_time column 'dt'")
        if not np.all(dt_numeric % 1 == 0):
            raise ValidationError("Fractional (non-integral) timestamps detected in event_time column 'dt'")

        df_work = df.copy()
        df_work["dt"] = dt_numeric.astype(np.int64)

        # 3. Auditable duplicate handling on (location_id, dt)
        dup_subset = ["location_id", "dt"] if "location_id" in df_work.columns else ["dt"]
        dup_mask = df_work.duplicated(subset=dup_subset, keep=False)
        identical_dups_dropped = 0

        if dup_mask.any():
            dup_rows = df_work[dup_mask]
            # Check if duplicates are identical across all 114 canonical features
            dup_features_mask = df_work.duplicated(subset=self.canonical_features, keep=False)
            conflicting = dup_rows[~dup_features_mask.loc[dup_rows.index]]
            if len(conflicting) > 0:
                conflicting_dts = list(conflicting["dt"].unique()[:5])
                raise ValidationError(
                    f"Conflicting duplicate records detected for entity key {dup_subset} "
                    f"at timestamps {conflicting_dts}. Failing validation."
                )

            # Identical duplicates: deterministically drop duplicates
            initial_count = len(df_work)
            df_work = df_work.drop_duplicates(subset=dup_subset, keep="first").reset_index(drop=True)
            identical_dups_dropped = initial_count - len(df_work)
            logger.info(f"Auditable deduplication: dropped {identical_dups_dropped} identical duplicate rows.")

        # 4. Numerical finiteness validation across all 114 canonical features
        feat_df = df_work[self.canonical_features]
        nan_count = int(feat_df.isna().sum().sum())
        if nan_count > 0:
            nan_cols = feat_df.columns[feat_df.isna().any()].tolist()
            raise ValidationError(f"NaN values detected in canonical features: {nan_cols[:5]}")

        inf_count = 0
        for col in self.canonical_features:
            if np.issubdtype(feat_df[col].dtype, np.number):
                inf_count += int(np.isinf(feat_df[col]).sum())
        if inf_count > 0:
            raise ValidationError(f"Infinite values detected in canonical features (count={inf_count}).")

        # 5. Sort strictly chronologically by dt
        df_sorted = df_work.sort_values("dt").reset_index(drop=True)

        audit_report = {
            "cleaned_row_count": len(df_sorted),
            "identical_duplicates_dropped": identical_dups_dropped,
            "earliest_dt": int(df_sorted["dt"].min()),
            "latest_dt": int(df_sorted["dt"].max()),
            "earliest_utc": pd.to_datetime(df_sorted["dt"].min(), unit="s", utc=True).isoformat(),
            "latest_utc": pd.to_datetime(df_sorted["dt"].max(), unit="s", utc=True).isoformat(),
        }
        return df_sorted, audit_report

    def filter_candidate_development_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Filter dataset to observations strictly before the protected holdout boundary.

        Candidate development is restricted to observations before 2025-06-07T00:00:00Z.
        The post-cutoff/quarantined region (observations on or after 2025-06-07) contains
        the formal protected 9,311-sample final test set (2025-06-07 through 2026-08-28)
        plus later accumulated observations, and is preserved completely untouched.

        Args:
            df: Cleaned chronological DataFrame.

        Returns:
            Tuple of (pre-holdout development DataFrame, holdout isolation metadata).
        """
        pre_holdout_mask = df["dt"] < self.HOLDOUT_START_DT
        df_dev = df[pre_holdout_mask].copy().reset_index(drop=True)
        post_cutoff_count = int((~pre_holdout_mask).sum())

        if len(df_dev) == 0:
            raise ValidationError("No observations found before protected holdout boundary (2025-06-07T00:00:00Z).")

        meta = {
            "holdout_start_dt": self.HOLDOUT_START_DT,
            "holdout_start_utc": pd.to_datetime(self.HOLDOUT_START_DT, unit="s", utc=True).isoformat(),
            "development_sample_count": len(df_dev),
            "protected_holdout_sample_count": post_cutoff_count,
            "post_cutoff_quarantined_sample_count": post_cutoff_count,
            "formal_protected_test_sample_count": 9311,
            "post_cutoff_quarantined_region_note": (
                "Contains formal 9,311-sample protected final test set (2025-06-07 to 2026-08-28) "
                "plus later accumulated observations."
            ),
            "development_earliest_dt": int(df_dev["dt"].min()),
            "development_latest_dt": int(df_dev["dt"].max()),
            "development_earliest_utc": pd.to_datetime(df_dev["dt"].min(), unit="s", utc=True).isoformat(),
            "development_latest_utc": pd.to_datetime(df_dev["dt"].max(), unit="s", utc=True).isoformat(),
            "holdout_preserved_untouched": True,
        }
        logger.info(
            f"Holdout boundary applied: {len(df_dev)} development samples (< 2025-06-07), "
            f"{post_cutoff_count} samples preserved in untouched post-cutoff/quarantined region."
        )
        return df_dev, meta

    def construct_physical_targets(
        self,
        df: pd.DataFrame,
        target_col: str = "epa_aqi",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Construct h1-h72 targets using exact physical timestamp semantics.

        For an observation at timestamp T, requires an actual observation at exactly
        T + h * 3600 seconds. Missing target timestamps cleanly drop the sample.
        Never interpolates target AQI and never relies on row position alone.

        Args:
            df: Input feature DataFrame with integer 'dt' column and target_col.
            target_col: Target AQI column name (default: 'epa_aqi').

        Returns:
            Tuple of (df_valid_features, df_valid_targets).
        """
        if target_col not in df.columns:
            raise ValidationError(f"Target column '{target_col}' not found in DataFrame.")

        dt_arr = df["dt"].astype(np.int64).values
        n_rows = len(df)
        ts_to_target = dict(zip(dt_arr, df[target_col].astype(np.float64)))

        target_matrix = np.full((n_rows, self.forecast_horizons), np.nan, dtype=np.float32)
        valid_mask = np.ones(n_rows, dtype=bool)

        for h in range(1, self.forecast_horizons + 1):
            step_seconds = h * 3600
            target_timestamps = dt_arr + step_seconds
            h_idx = h - 1
            for i, target_ts in enumerate(target_timestamps):
                if not valid_mask[i]:
                    continue
                val = ts_to_target.get(target_ts)
                if val is None or np.isnan(val):
                    valid_mask[i] = False
                else:
                    target_matrix[i, h_idx] = val

        valid_indices = np.where(valid_mask)[0]
        df_valid_features = df.iloc[valid_indices].copy().reset_index(drop=True)
        target_cols = [f"target_h{h}" for h in range(1, self.forecast_horizons + 1)]
        df_valid_targets = pd.DataFrame(target_matrix[valid_indices], columns=target_cols)

        dropped_count = n_rows - len(df_valid_features)
        logger.info(
            f"Physical target construction ({self.forecast_horizons}h): {len(df_valid_features)} valid samples "
            f"retained (dropped {dropped_count} samples missing exact physical target timestamps)."
        )
        return df_valid_features, df_valid_targets

    def split_chronological_with_embargo(
        self,
        df_features: pd.DataFrame,
        df_targets: pd.DataFrame,
        train_ratio: float = 0.8,
    ) -> dict[str, Any]:
        """Perform chronological train/val split with strict anti-leakage embargo.

        Enforces:
            min(val_input_dt) > max(train_input_dt) + 72h
            max_train_target_dt < min_val_input_dt

        Args:
            df_features: Feature DataFrame with 'dt' column.
            df_targets: Multi-output target DataFrame.
            train_ratio: Chronological train split fraction (default: 0.80).

        Returns:
            Dictionary containing train and val splits and leakage audit metadata.
        """
        total_samples = len(df_features)
        split_idx = int(total_samples * train_ratio)
        split_dt = int(df_features["dt"].iloc[split_idx])

        # Embargo gap: last allowed training sample timestamp must be strictly before split_dt - 72h
        # On an hourly grid this requires at least a 73-hour separation between final train input and first val input
        embargo_seconds = self.forecast_horizons * 3600
        train_cutoff_dt = split_dt - embargo_seconds - 3600

        train_mask = df_features["dt"] <= train_cutoff_dt
        val_mask = df_features["dt"] >= split_dt

        df_train_feats = df_features.loc[train_mask].copy().reset_index(drop=True)
        df_train_targets = df_targets.loc[train_mask].copy().reset_index(drop=True)

        df_val_feats = df_features.loc[val_mask].copy().reset_index(drop=True)
        df_val_targets = df_targets.loc[val_mask].copy().reset_index(drop=True)

        max_train_input_dt = int(df_train_feats["dt"].max())
        min_val_input_dt = int(df_val_feats["dt"].min())
        max_train_target_dt = max_train_input_dt + embargo_seconds

        # Strict Leakage Audit Checks
        if not (min_val_input_dt > max_train_input_dt + embargo_seconds):
            raise ValidationError(
                f"Embargo rule violated: min_val_input_dt ({min_val_input_dt}) <= "
                f"max_train_input_dt + 72h ({max_train_input_dt + embargo_seconds})"
            )

        if not (max_train_target_dt < min_val_input_dt):
            raise ValidationError(
                f"Temporal leakage detected: max_train_target_dt ({max_train_target_dt}) >= "
                f"min_val_input_dt ({min_val_input_dt})"
            )

        separation_hours = round((min_val_input_dt - max_train_input_dt) / 3600.0, 2)
        embargoed_samples = total_samples - (len(df_train_feats) + len(df_val_feats))

        leakage_audit = {
            "max_train_input_dt": max_train_input_dt,
            "min_val_input_dt": min_val_input_dt,
            "max_train_target_dt": max_train_target_dt,
            "separation_hours": separation_hours,
            "embargo_hours_required": self.forecast_horizons,
            "leakage_rule_passed": True,
            "max_train_target_lt_min_val_input": True,
            "min_val_gt_max_train_plus_72h": True,
            "embargoed_samples_count": embargoed_samples,
            "train_sample_count": len(df_train_feats),
            "val_sample_count": len(df_val_feats),
            "train_period": {
                "start_utc": pd.to_datetime(df_train_feats["dt"].min(), unit="s", utc=True).isoformat(),
                "end_utc": pd.to_datetime(max_train_input_dt, unit="s", utc=True).isoformat(),
            },
            "val_period": {
                "start_utc": pd.to_datetime(min_val_input_dt, unit="s", utc=True).isoformat(),
                "end_utc": pd.to_datetime(df_val_feats["dt"].max(), unit="s", utc=True).isoformat(),
            },
        }

        logger.info(
            f"Leakage audit PASSED: train_end={leakage_audit['train_period']['end_utc']}, "
            f"val_start={leakage_audit['val_period']['start_utc']} "
            f"({separation_hours}h separation, {embargoed_samples} embargoed samples)."
        )

        return {
            "X_train_df": df_train_feats,
            "Y_train_df": df_train_targets,
            "X_val_df": df_val_feats,
            "Y_val_df": df_val_targets,
            "leakage_audit": leakage_audit,
        }

