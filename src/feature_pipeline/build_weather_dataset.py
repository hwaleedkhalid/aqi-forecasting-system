"""Pearls AQI Predictor - Weather-Enriched Dataset Builder.

Coordinates Open-Meteo weather data ingestion, weather feature engineering,
temporal fusion with pollutant features, and multi-output train/test dataset creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import (
    FORECAST_HORIZONS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_WEATHER_DIR,
    TRAIN_TEST_SPLIT_RATIO,
)
from src.data_ingestion.weather_provider import OpenMeteoWeatherProvider
from src.feature_pipeline.feature_engineering import FeatureEngineeringPipeline
from src.feature_pipeline.weather_features import WeatherFeatureEngineer
from src.logger import logger


def run_weather_enrichment_pipeline(
    data_dir: Path = PROCESSED_DATA_DIR,
    raw_weather_dir: Path = RAW_WEATHER_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Execute complete weather data ingestion, feature fusion, and dataset construction.

    Returns:
        Summary dictionary of generated datasets and shapes.
    """
    logger.info("=== STEP 1: Fetching historical weather from Open-Meteo ===")
    provider = OpenMeteoWeatherProvider(raw_dir=raw_weather_dir)
    df_raw_weather = provider.fetch_historical_weather(
        start_date="2020-11-27",
        end_date="2026-08-31",
        cache_filename="openmeteo_lahore_historical.json",
    )

    logger.info("=== STEP 2: Engineering meteorological features ===")
    engineer = WeatherFeatureEngineer()
    df_weather_features = engineer.build_weather_features(df_raw_weather)

    logger.info("=== STEP 3: Loading baseline pollutant features ===")
    pollutants_path = data_dir / "features.csv"
    if not pollutants_path.exists():
        logger.info("features.csv not found, rebuilding from clean historical AQI...")
        fe_pipe = FeatureEngineeringPipeline()
        df_clean = pd.read_csv(data_dir / "historical_aqi_clean.csv")
        df_pollutants, _ = fe_pipe.build_features(df_clean)
        df_pollutants = df_pollutants.reset_index()
    else:
        df_pollutants = pd.read_csv(pollutants_path)

    logger.info("=== STEP 4: Fusing pollutants and weather features ===")
    df_enriched = engineer.fuse_features(df_pollutants, df_weather_features, drop_na=True)

    # Save enriched feature matrix and schema
    enriched_csv_path = data_dir / "features_v2_weather.csv"
    df_enriched.to_csv(enriched_csv_path, index=False)
    logger.info(f"Saved enriched feature dataset to {enriched_csv_path}")

    schema_path = data_dir / "feature_schema_v2_weather.json"
    engineer.save_enriched_schema(schema_path)

    logger.info("=== STEP 5: Building multi-output dataset with 72h embargo (Vectorized) ===")
    # Target matrix Y = [AQI(t+1), ..., AQI(t+72)]
    # Target is generated from continuous hourly grid
    df_clean = pd.read_csv(data_dir / "historical_aqi_clean.csv")
    df_clean["datetime_utc"] = pd.to_datetime(df_clean["datetime_utc"], utc=True)
    df_clean = df_clean.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"]).reset_index(drop=True)
    
    # Reindex to continuous 1h grid
    full_grid = pd.DataFrame({
        "datetime_utc": pd.date_range(
            df_clean["datetime_utc"].min(), df_clean["datetime_utc"].max(), freq="1h", tz="UTC"
        )
    })
    df_grid = pd.merge(full_grid, df_clean[["datetime_utc", "epa_aqi"]], on="datetime_utc", how="left")
    
    # Vectorized future targets
    for h in range(1, FORECAST_HORIZONS + 1):
        df_grid[f"target_h{h}"] = df_grid["epa_aqi"].shift(-h)
    
    # Merge targets back into enriched features (avoiding duplicate epa_aqi columns)
    df_enriched["datetime_utc"] = pd.to_datetime(df_enriched["datetime_utc"], utc=True)
    target_cols = [f"target_h{h}" for h in range(1, FORECAST_HORIZONS + 1)]
    merged_data = pd.merge(df_enriched, df_grid[["datetime_utc"] + target_cols], on="datetime_utc", how="inner")
    
    # Drop rows where any future target or feature is NaN
    merged_data = merged_data.dropna().reset_index(drop=True)
    
    timestamps = merged_data["datetime_utc"]
    feature_cols = engineer.feature_schema
    X_mat_raw = merged_data[feature_cols].values.astype(np.float32)
    Y_mat = merged_data[target_cols].values.astype(np.float32)
    current_aqi_arr = merged_data["epa_aqi"].values.astype(np.float32)

    total_pairs = len(merged_data)
    logger.info(f"Constructed {total_pairs} supervised (X, Y_72h) pairs with {len(feature_cols)} features.")

    # 80/20 Chronological Split with 72h Embargo
    split_index = int(total_pairs * TRAIN_TEST_SPLIT_RATIO)
    split_ts = timestamps.iloc[split_index]
    embargo_cutoff = split_ts - pd.Timedelta(hours=FORECAST_HORIZONS)

    train_mask = timestamps <= embargo_cutoff
    test_mask = timestamps >= split_ts
    embargo_mask = (timestamps > embargo_cutoff) & (timestamps < split_ts)

    X_train_raw = X_mat_raw[train_mask]
    y_train = Y_mat[train_mask]
    current_aqi_train = current_aqi_arr[train_mask]

    X_test_raw = X_mat_raw[test_mask]
    y_test = Y_mat[test_mask]
    current_aqi_test = current_aqi_arr[test_mask]

    # Standard Scaler fit strictly on training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Save scaler artifact
    scaler_path = models_dir / "feature_scaler_v2_weather.joblib"
    joblib.dump(scaler, scaler_path)

    # Save numpy dataset arrays
    np.save(data_dir / "X_train_v2.npy", X_train)
    np.save(data_dir / "y_train_v2.npy", y_train)
    np.save(data_dir / "X_test_v2.npy", X_test)
    np.save(data_dir / "y_test_v2.npy", y_test)
    np.save(data_dir / "current_aqi_train_v2.npy", current_aqi_train)
    np.save(data_dir / "current_aqi_test_v2.npy", current_aqi_test)

    # Save summary metadata
    summary = {
        "version": "v2_weather_enriched",
        "total_features": len(feature_cols),
        "feature_names": feature_cols,
        "total_supervised_pairs": total_pairs,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "embargoed_samples": int(embargo_mask.sum()),
        "train_period": {
            "start_utc": str(timestamps[train_mask].min()),
            "end_utc": str(timestamps[train_mask].max()),
        },
        "test_period": {
            "start_utc": str(timestamps[test_mask].min()),
            "end_utc": str(timestamps[test_mask].max()),
        },
        "split_timestamp_utc": str(split_ts),
        "embargo_gap_hours": FORECAST_HORIZONS,
    }

    with open(data_dir / "dataset_summary_v2_weather.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Enriched dataset successfully created: Train={X_train.shape}, Test={X_test.shape}, "
        f"Embargoed={embargo_mask.sum()} samples."
    )
    return summary


if __name__ == "__main__":
    run_weather_enrichment_pipeline()
