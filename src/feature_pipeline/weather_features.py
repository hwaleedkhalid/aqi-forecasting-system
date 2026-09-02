"""Pearls AQI Predictor - Weather Feature Engineering & Fusion Pipeline.

Processes raw meteorological telemetry, generates temporal lags, rolling statistics,
wind vector components, and atmospheric dispersion interaction terms, fusing them
with air quality features under strict timestamp alignment.
"""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.config import LAG_HOURS, PROCESSED_DATA_DIR, ROLLING_WINDOWS
from src.exceptions import FeatureStoreError, ValidationError
from src.logger import logger


class WeatherFeatureEngineer:
    """Computes meteorological features and performs leakage-free temporal joins with air quality."""

    WEATHER_COLUMNS: list[str] = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
    ]

    def __init__(
        self,
        lag_hours: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ) -> None:
        """Initialize WeatherFeatureEngineer.

        Args:
            lag_hours: Forecast lag offsets in hours (default: [1, 3, 6, 12, 24]).
            rolling_windows: Rolling aggregation windows in hours (default: [6, 12, 24]).
        """
        self.lag_hours = lag_hours or LAG_HOURS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.feature_schema: list[str] = []

    def build_weather_features(self, df_weather: pd.DataFrame) -> pd.DataFrame:
        """Engineer meteorological features on a continuous hourly time series.

        Args:
            df_weather: Raw weather dataframe containing datetime_utc and WEATHER_COLUMNS.

        Returns:
            DataFrame containing engineered weather features indexed by datetime_utc.
        """
        df = df_weather.copy()
        if "datetime_utc" not in df.columns:
            raise ValidationError("Input weather dataframe must contain 'datetime_utc' column.")

        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
        df = df.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"]).reset_index(drop=True)

        # 1. Reindex to continuous 1h UTC frequency to guarantee lag integrity
        df = df.set_index("datetime_utc").asfreq("1h")
        for col in self.WEATHER_COLUMNS:
            if col in df.columns:
                df[col] = df[col].interpolate(method="time", limit=3)

        df = df.reset_index()

        # 2. Wind Direction Trigonometric Vector Components
        wind_rad = np.radians(df["wind_direction_10m"].values)
        df["wind_dir_sin"] = np.sin(wind_rad)
        df["wind_dir_cos"] = np.cos(wind_rad)

        # 3. Barometric Pressure Differentials (Pressure Tendency)
        df["pressure_diff_1h"] = df["surface_pressure"] - df["surface_pressure"].shift(1)
        df["pressure_diff_24h"] = df["surface_pressure"] - df["surface_pressure"].shift(24)

        # 4. Thermal Moisture Index
        df["thermal_moisture_index"] = df["temperature_2m"] * (df["relative_humidity_2m"] / 100.0)

        # 5. Backward-looking Weather Lags
        lag_vars = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure"]
        for var in lag_vars:
            for lag in self.lag_hours:
                df[f"{var}_lag_{lag}h"] = df[var].shift(lag)

        # 6. Backward-looking Rolling Statistics (mean and std)
        rolling_vars = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
        for var in rolling_vars:
            for window in self.rolling_windows:
                # min_periods=1 ensures no arbitrary dropped rows during partial periods
                df[f"{var}_rolling_mean_{window}h"] = df[var].rolling(window=window, min_periods=window // 2).mean()
                df[f"{var}_rolling_std_{window}h"] = (
                    df[var].rolling(window=window, min_periods=window // 2).std().fillna(0.0)
                )

        # Drop raw non-decomposed wind direction degrees
        df = df.drop(columns=["wind_direction_10m"])
        return df

    def fuse_features(
        self,
        df_pollutants: pd.DataFrame,
        df_weather_features: pd.DataFrame,
        drop_na: bool = True,
    ) -> pd.DataFrame:
        """Perform exact temporal inner join between pollutant features and weather features.

        Args:
            df_pollutants: Feature dataframe from Phase 6 (containing epa_aqi, pm2_5, lags, etc.).
            df_weather_features: Engineered weather features from build_weather_features.
            drop_na: If True, drops rows with missing lag/rolling warm-up periods.

        Returns:
            Unified feature dataframe containing pollutants, weather, and dispersion interactions.
        """
        df_pol = df_pollutants.copy()
        df_wea = df_weather_features.copy()

        df_pol["datetime_utc"] = pd.to_datetime(df_pol["datetime_utc"], utc=True)
        df_wea["datetime_utc"] = pd.to_datetime(df_wea["datetime_utc"], utc=True)

        logger.info(
            f"Fusing pollutant features ({len(df_pol)} rows) and weather features ({len(df_wea)} rows) on datetime_utc..."
        )

        merged = pd.merge(df_pol, df_wea, on="datetime_utc", how="inner")

        # 7. Atmospheric Stagnation Dispersion Index (pm2_5 / (wind_speed + 0.5))
        # Strictly uses concurrent observations at timestamp t
        if "pm2_5" in merged.columns and "wind_speed_10m" in merged.columns:
            merged["stagnation_index"] = merged["pm2_5"] / (merged["wind_speed_10m"] + 0.5)

        if drop_na:
            init_len = len(merged)
            merged = merged.dropna().reset_index(drop=True)
            logger.info(f"Dropped {init_len - len(merged)} warm-up/gap rows, retained {len(merged)} complete rows.")

        # Save numeric feature column names (excluding timestamps and non-numeric labels)
        feature_cols = [
            c for c in merged.select_dtypes(include=[np.number]).columns
            if c not in ["datetime_utc", "dominant_pollutant", "aqi_category"]
        ]
        self.feature_schema = sorted(feature_cols)

        logger.info(f"Feature fusion complete: {len(merged)} rows x {len(self.feature_schema)} features.")
        return merged

    def save_enriched_schema(
        self,
        output_path: Path = PROCESSED_DATA_DIR / "feature_schema_v2_weather.json",
    ) -> Path:
        """Serialize enriched feature schema to JSON.

        Args:
            output_path: Path to destination schema JSON.

        Returns:
            Path to saved schema artifact.
        """
        if not self.feature_schema:
            raise FeatureStoreError("Feature schema is empty. Run fuse_features() first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        schema_dict = {
            "version": "v2_weather_enriched",
            "total_features": len(self.feature_schema),
            "feature_names": self.feature_schema,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)

        logger.info(f"Saved enriched feature schema ({len(self.feature_schema)} features) to {output_path}")
        return output_path
