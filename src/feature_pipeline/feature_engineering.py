"""Pearls AQI Predictor - Feature Engineering Pipeline.

Constructs strictly backward-looking feature representations from historical
pollutant time series without future data leakage. Generates temporal cyclical
signals, historical lags, backward rolling aggregates, rate of change indicators,
and zero-division-safe chemical interaction ratios.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.config import (
    LAG_HOURS,
    POLLUTANTS,
    PROCESSED_DATA_DIR,
    ROLLING_WINDOWS,
)
from src.exceptions import ValidationError
from src.logger import logger

# Columns to generate lag features for
LAG_COLUMNS = ["pm2_5", "pm10", "no2", "o3", "epa_aqi"]

# Columns to generate rolling window aggregates for
ROLLING_COLUMNS = ["pm2_5", "epa_aqi"]


class FeatureEngineeringPipeline:
    """End-to-end feature transformation pipeline for air quality time series."""

    def __init__(
        self,
        lag_hours: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        """Initialize FeatureEngineeringPipeline.

        Args:
            lag_hours: List of hourly lag steps. Defaults to LAG_HOURS config.
            rolling_windows: List of rolling window sizes in hours. Defaults to ROLLING_WINDOWS.
        """
        self.lag_hours = sorted(lag_hours if lag_hours is not None else LAG_HOURS)
        self.rolling_windows = sorted(
            rolling_windows if rolling_windows is not None else ROLLING_WINDOWS
        )
        self.feature_columns: list[str] = []

    def prepare_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure continuous hourly index and handle short sensor dropouts.

        Reindexes the DataFrame to an exact 1-hour grid so that shift operations
        represent physical temporal steps. Linearly interpolates short missing
        gaps (<= 3 hours).

        Args:
            df: Input DataFrame containing 'datetime_utc' or 'dt' column.

        Returns:
            Clean continuous hourly DataFrame indexed by UTC datetime.
        """
        df_ts = df.copy()

        if "datetime_utc" not in df_ts.columns:
            if "dt" in df_ts.columns:
                df_ts["datetime_utc"] = pd.to_datetime(df_ts["dt"], unit="s", utc=True)
            else:
                raise ValidationError(
                    "DataFrame must contain 'datetime_utc' or 'dt' timestamp column"
                )
        else:
            df_ts["datetime_utc"] = pd.to_datetime(df_ts["datetime_utc"], utc=True)

        # Sort and deduplicate timestamps
        df_ts = df_ts.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc")
        df_ts = df_ts.set_index("datetime_utc")

        # Reindex to full hourly grid
        full_idx = pd.date_range(
            start=df_ts.index.min(),
            end=df_ts.index.max(),
            freq="1h",
            name="datetime_utc",
        )
        df_reindexed = df_ts.reindex(full_idx)

        # Interpolate short gaps (<= 3 hours) for continuous feature continuity
        numeric_cols = df_reindexed.select_dtypes(include=[np.number]).columns
        df_reindexed[numeric_cols] = df_reindexed[numeric_cols].interpolate(
            method="linear", limit=3
        )

        return df_reindexed

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create cyclical temporal representations available at inference time.

        Calculates sin/cos transformations for hour-of-day, day-of-week, and month.

        Args:
            df: Datetime-indexed DataFrame.

        Returns:
            DataFrame augmented with temporal feature columns.
        """
        df_out = df.copy()
        idx = df_out.index

        # Hour of day (0-23)
        hour = idx.hour
        df_out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df_out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Day of week (0=Monday, 6=Sunday)
        day_of_week = idx.dayofweek
        df_out["day_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
        df_out["day_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

        # Month of year (1-12)
        month = idx.month
        df_out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
        df_out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

        # Weekend indicator
        df_out["is_weekend"] = (day_of_week >= 5).astype(int)

        return df_out

    def create_lag_features(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Generate strictly historical backward lag features.

        Args:
            df: Continuous DatetimeIndex DataFrame.
            columns: List of columns to lag. Defaults to LAG_COLUMNS.

        Returns:
            DataFrame with lag columns appended.
        """
        df_out = df.copy()
        target_cols = columns or [c for c in LAG_COLUMNS if c in df_out.columns]

        for col in target_cols:
            for lag in self.lag_hours:
                col_name = f"{col}_lag_{lag}h"
                # shift(k) moves past observations into current row (strictly backward looking)
                df_out[col_name] = df_out[col].shift(lag)

        return df_out

    def create_rolling_features(
        self, df: pd.DataFrame, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Generate backward rolling statistical aggregates.

        Uses standard Pandas closed='right' rolling windows that include only
        the current observation and past observations [t-w+1 ... t].

        Args:
            df: Continuous DatetimeIndex DataFrame.
            columns: List of columns to aggregate. Defaults to ROLLING_COLUMNS.

        Returns:
            DataFrame with rolling features appended.
        """
        df_out = df.copy()
        target_cols = columns or [c for c in ROLLING_COLUMNS if c in df_out.columns]

        for col in target_cols:
            for w in self.rolling_windows:
                # Mean over past w hours
                df_out[f"{col}_rolling_mean_{w}h"] = (
                    df_out[col].rolling(window=w, min_periods=max(1, w // 2)).mean()
                )
                # Standard deviation (volatility) over past w hours
                df_out[f"{col}_rolling_std_{w}h"] = (
                    df_out[col].rolling(window=w, min_periods=max(1, w // 2)).std().fillna(0.0)
                )

            # Min and Max over 24-hour window
            df_out[f"{col}_rolling_min_24h"] = (
                df_out[col].rolling(window=24, min_periods=12).min()
            )
            df_out[f"{col}_rolling_max_24h"] = (
                df_out[col].rolling(window=24, min_periods=12).max()
            )

        return df_out

    def create_ratio_and_diff_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create chemical interaction ratios and short-term differential signals.

        Safely handles division by zero using small epsilon term.

        Args:
            df: DataFrame containing base and lag features.

        Returns:
            DataFrame with ratio and difference features.
        """
        df_out = df.copy()
        eps = 1e-5

        # Chemical Ratios
        if "pm2_5" in df_out.columns and "pm10" in df_out.columns:
            # High PM2.5/PM10 ratio indicates fine combustion/smog dominance
            df_out["pm_ratio"] = df_out["pm2_5"] / (df_out["pm10"] + eps)

        if "no2" in df_out.columns and "o3" in df_out.columns:
            # NO2/O3 ratio indicates photochemical oxidation state
            df_out["nitrogen_ozone_ratio"] = df_out["no2"] / (df_out["o3"] + eps)

        if "co" in df_out.columns and "no2" in df_out.columns:
            # CO/NO2 indicates combustion efficiency / vehicle emissions
            df_out["combustion_index"] = df_out["co"] / (df_out["no2"] + eps)

        # Rate of Change (Differential) Features
        if "pm2_5" in df_out.columns:
            df_out["pm2_5_diff_1h"] = df_out["pm2_5"] - df_out["pm2_5"].shift(1)
            df_out["pm2_5_diff_24h"] = df_out["pm2_5"] - df_out["pm2_5"].shift(24)

        if "epa_aqi" in df_out.columns:
            df_out["epa_aqi_diff_1h"] = df_out["epa_aqi"] - df_out["epa_aqi"].shift(1)
            df_out["epa_aqi_diff_24h"] = df_out["epa_aqi"] - df_out["epa_aqi"].shift(24)

        return df_out

    def build_features(
        self, df: pd.DataFrame, drop_na: bool = True
    ) -> tuple[pd.DataFrame, list[str]]:
        """Execute full feature engineering pipeline.

        Generates all feature groups in deterministic column order.

        Args:
            df: Input raw or clean DataFrame.
            drop_na: If True, drop the warm-up rows (first 24h) containing NaNs.

        Returns:
            Tuple of (features_dataframe, ordered_feature_names_list).
        """
        logger.info("Executing feature engineering pipeline...")
        df_ts = self.prepare_time_series(df)

        df_feat = self.create_temporal_features(df_ts)
        df_feat = self.create_lag_features(df_feat)
        df_feat = self.create_rolling_features(df_feat)
        df_feat = self.create_ratio_and_diff_features(df_feat)

        # Identify candidate feature columns (exclude raw non-numeric metadata)
        exclude_cols = {
            "dt",
            "datetime_utc",
            "owm_caqi",
            "dominant_pollutant",
            "aqi_category",
            "aqi_color",
            "sub_aqi_pm2_5",
            "sub_aqi_pm10",
            "sub_aqi_o3",
            "sub_aqi_no2",
            "sub_aqi_so2",
            "sub_aqi_co",
        }

        feature_cols = [c for c in df_feat.columns if c not in exclude_cols]
        # Sort feature columns deterministically
        feature_cols = sorted(feature_cols)

        if drop_na:
            initial_len = len(df_feat)
            df_feat = df_feat.dropna(subset=feature_cols)
            dropped = initial_len - len(df_feat)
            logger.info(
                f"Dropped {dropped} initialization rows containing lag/rolling NaNs. "
                f"Remaining samples: {len(df_feat):,}"
            )

        self.feature_columns = feature_cols
        logger.info(f"Successfully engineered {len(feature_cols)} features.")

        return df_feat, feature_cols

    def save_feature_schema(
        self, output_path: Path = PROCESSED_DATA_DIR / "feature_schema.json"
    ) -> Path:
        """Persist feature schema and deterministic column order to JSON.

        Args:
            output_path: Destination JSON file path.

        Returns:
            Path to saved feature schema file.
        """
        if not self.feature_columns:
            raise ValidationError(
                "Feature columns list is empty. Call build_features() first."
            )

        schema = {
            "total_features": len(self.feature_columns),
            "feature_names": self.feature_columns,
            "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "lag_hours": self.lag_hours,
            "rolling_windows": self.rolling_windows,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)

        logger.info(f"Saved feature schema ({len(self.feature_columns)} features) to {output_path}")
        return output_path
