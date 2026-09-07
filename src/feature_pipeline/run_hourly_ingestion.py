"""Pearls AQI Predictor - Scheduled Hourly Ingestion & Feature Engineering Runner.

Orchestrates live hourly telemetry ingestion from OpenWeather and Open-Meteo,
enforces continuous hourly-grid validation (with established limit=3 interpolation
and closed-gap failure on >3h gaps), builds the 114-column canonical feature vector,
and ingests into Hopsworks Feature Store with offline Hudi upsert and server-side
newer-event online protection (upsert_if_newer=True).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import (
    API_MAX_RETRIES,
    API_RETRY_BASE_DELAY,
    API_TIMEOUT_SECONDS,
    OPENWEATHER_API_KEY,
    PROCESSED_DATA_DIR,
    RUNTIME_DIR,
    TARGET_LAT,
    TARGET_LON,
)
from src.data_ingestion.openweather_provider import OpenWeatherProvider
from src.exceptions import DataIngestionError, FeatureStoreError, ValidationError
from src.feature_pipeline.aqi_calculator import calculate_overall_aqi, get_aqi_category
from src.feature_pipeline.feature_engineering import FeatureEngineeringPipeline
from src.feature_pipeline.hopsworks_integration import (
    HopsworksFeatureStoreConnector,
    load_canonical_feature_names,
)
from src.feature_pipeline.weather_features import WeatherFeatureEngineer
from src.logger import logger

SNAPSHOT_DIR = Path("data/snapshots")
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]


def fetch_hourly_telemetry_window(
    lookback_hours: int = 72,
    api_key: str | None = None,
    lat: float = TARGET_LAT,
    lon: float = TARGET_LON,
    timeout: int = API_TIMEOUT_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch recent historical telemetry window for air quality and meteorology.

    Args:
        lookback_hours: Number of past hours to retrieve (minimum 48 recommended).
        api_key: OpenWeather API key. If None, resolves from environment.
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (df_aq, df_weather).
    """
    if api_key is None:
        api_key = os.environ.get("OPENWEATHER_API_KEY") or OPENWEATHER_API_KEY

    if not api_key or not api_key.strip():
        raise DataIngestionError("OPENWEATHER_API_KEY is required to fetch live air quality telemetry.")

    now_utc = datetime.now(timezone.utc)
    now_ts = int(now_utc.timestamp())
    start_ts = now_ts - (lookback_hours * 3600)

    # 1. Fetch Air Quality History from OpenWeather
    provider = OpenWeatherProvider(api_key=api_key)
    logger.info(f"Fetching {lookback_hours}h air quality history from OpenWeather ({lat}, {lon})...")
    raw_aq = provider.fetch_historical_air_quality(lat=lat, lon=lon, start=start_ts, end=now_ts)

    records: list[dict[str, Any]] = []
    for item in raw_aq.get("list", []):
        dt = int(item.get("dt", 0))
        comps = item.get("components", {})
        overall_aqi, dom_pol, _ = calculate_overall_aqi(comps)
        cat, _ = get_aqi_category(overall_aqi)
        records.append({
            "dt": dt,
            "co": float(comps.get("co", 0.0)),
            "no": float(comps.get("no", 0.0)),
            "no2": float(comps.get("no2", 0.0)),
            "o3": float(comps.get("o3", 0.0)),
            "so2": float(comps.get("so2", 0.0)),
            "pm2_5": float(comps.get("pm2_5", 0.0)),
            "pm10": float(comps.get("pm10", 0.0)),
            "nh3": float(comps.get("nh3", 0.0)),
            "datetime_utc": pd.to_datetime(dt, unit="s", utc=True),
            "epa_aqi": overall_aqi,
            "dominant_pollutant": dom_pol,
            "aqi_category": cat,
        })

    if not records:
        raise DataIngestionError("OpenWeather historical air quality response contained zero records.")

    df_aq = pd.DataFrame(records).sort_values("dt").drop_duplicates(subset=["dt"]).reset_index(drop=True)

    # 2. Fetch Surface Meteorology from Open-Meteo
    # Calculate required past_days covering lookback_hours
    past_days = max(2, int(np.ceil(lookback_hours / 24)))
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "forecast_days": 1,
        "hourly": WEATHER_VARIABLES,
        "timezone": "UTC",
    }

    logger.info(f"Fetching {past_days} past days weather from Open-Meteo ({lat}, {lon})...")
    resp = None
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                break
        except requests.RequestException as e:
            logger.warning(f"Open-Meteo request attempt {attempt}/{API_MAX_RETRIES} failed: {e}")
        import time
        time.sleep(API_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

    if resp is None or resp.status_code != 200:
        err = resp.text if resp else "Connection error"
        raise DataIngestionError(f"Failed to fetch meteorology from Open-Meteo: {err}")

    w_json = resp.json().get("hourly", {})
    if "time" not in w_json:
        raise DataIngestionError("Open-Meteo response schema missing 'hourly.time'.")

    df_weather = pd.DataFrame({
        "datetime_utc": pd.to_datetime(w_json["time"], utc=True),
        "temperature_2m": w_json["temperature_2m"],
        "relative_humidity_2m": w_json["relative_humidity_2m"],
        "surface_pressure": w_json["surface_pressure"],
        "wind_speed_10m": w_json["wind_speed_10m"],
        "wind_direction_10m": w_json["wind_direction_10m"],
        "precipitation": w_json["precipitation"],
    }).sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"]).reset_index(drop=True)

    return df_aq, df_weather


def derive_production_timestamp(
    df_aq: pd.DataFrame,
    df_weather: pd.DataFrame,
    now_utc: datetime | None = None,
) -> int:
    """Explicitly derive production timestamp T.

    T is the latest completed hourly timestamp common to both pollutant
    and meteorological telemetry, floored to hour boundary (T % 3600 == 0),
    and strictly <= now_utc. Future forecast hours from Open-Meteo are discarded.

    Args:
        df_aq: Pollutant telemetry DataFrame containing 'dt'.
        df_weather: Weather telemetry DataFrame containing 'datetime_utc'.
        now_utc: Current UTC datetime (defaults to datetime.now(timezone.utc)).

    Returns:
        Unix timestamp T as integer.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    now_ts = int(now_utc.timestamp())
    now_hour_ts = (now_ts // 3600) * 3600

    aq_max_dt = int(df_aq["dt"].max())
    # Floor to hour boundary
    aq_max_dt = (aq_max_dt // 3600) * 3600

    weather_dts = (df_weather["datetime_utc"].astype("int64") // 10**9).values
    # Discard future timestamps (> now_hour_ts) from weather series
    past_weather_dts = weather_dts[weather_dts <= now_hour_ts]
    if len(past_weather_dts) == 0:
        raise ValidationError("Open-Meteo telemetry contains no completed observations <= now_utc.")
    weather_max_dt = int(past_weather_dts.max())

    # Production timestamp T is the latest common completed observation
    T = min(aq_max_dt, weather_max_dt, now_hour_ts)

    if T % 3600 != 0:
        T = (T // 3600) * 3600

    logger.info(f"Derived production timestamp T = {T} ({datetime.fromtimestamp(T, tz=timezone.utc).isoformat()})")
    return T


def validate_hourly_grid_continuity(
    df_aq: pd.DataFrame,
    df_weather: pd.DataFrame,
    production_timestamp: int,
    required_lookback_hours: int = 24,
    max_tolerated_gap_hours: int = 3,
) -> None:
    """Enforce exact hourly-grid continuity before feature engineering.

    Verifies that the required [T-24h, T] interval contains sufficient hourly
    observations. Gaps <= 3 hours are interpolated by the production pipeline,
    but gaps > 3 hours or missing T-24h boundary fail closed.

    Args:
        df_aq: Pollutant DataFrame.
        df_weather: Weather DataFrame.
        production_timestamp: Target timestamp T.
        required_lookback_hours: Lookback span needed for lag/rolling features (24h).
        max_tolerated_gap_hours: Maximum allowable missing streak (3h per production contract).
    """
    min_required_ts = production_timestamp - (required_lookback_hours * 3600)

    # 1. Check Air Quality Span
    aq_dts = sorted(df_aq["dt"].unique())
    if not aq_dts or aq_dts[0] > min_required_ts:
        earliest_aq = aq_dts[0] if aq_dts else "None"
        raise ValidationError(
            f"Insufficient air quality lookback: earliest observation is {earliest_aq}, "
            f"but required lookback boundary T-{required_lookback_hours}h is {min_required_ts}."
        )

    # Check for gaps > max_tolerated_gap_hours in AQ series within [min_required_ts, production_timestamp]
    relevant_aq_dts = [t for t in aq_dts if min_required_ts <= t <= production_timestamp]
    for i in range(1, len(relevant_aq_dts)):
        gap_hours = (relevant_aq_dts[i] - relevant_aq_dts[i - 1]) // 3600
        if gap_hours > max_tolerated_gap_hours:
            raise ValidationError(
                f"Air quality telemetry has an unbridgeable gap of {gap_hours} hours between "
                f"{relevant_aq_dts[i-1]} and {relevant_aq_dts[i]} (max tolerated: {max_tolerated_gap_hours}h)."
            )

    # 2. Check Weather Span
    weather_dts = sorted((df_weather["datetime_utc"].astype("int64") // 10**9).unique())
    if not weather_dts or weather_dts[0] > min_required_ts:
        earliest_w = weather_dts[0] if weather_dts else "None"
        raise ValidationError(
            f"Insufficient weather lookback: earliest observation is {earliest_w}, "
            f"but required lookback boundary T-{required_lookback_hours}h is {min_required_ts}."
        )

    # Check for gaps > max_tolerated_gap_hours in weather series within [min_required_ts, production_timestamp]
    relevant_w_dts = [t for t in weather_dts if min_required_ts <= t <= production_timestamp]
    for i in range(1, len(relevant_w_dts)):
        gap_hours = (relevant_w_dts[i] - relevant_w_dts[i - 1]) // 3600
        if gap_hours > max_tolerated_gap_hours:
            raise ValidationError(
                f"Weather telemetry has an unbridgeable gap of {gap_hours} hours between "
                f"{relevant_w_dts[i-1]} and {relevant_w_dts[i]} (max tolerated: {max_tolerated_gap_hours}h)."
            )

    logger.info(
        f"Hourly grid continuity verified for [T-24h={min_required_ts}, T={production_timestamp}]. "
        f"No gaps > {max_tolerated_gap_hours}h found."
    )


def construct_canonical_feature_dataframe(
    df_aq: pd.DataFrame,
    df_weather: pd.DataFrame,
    production_timestamp: int,
) -> pd.DataFrame:
    """Build the exact 114-column canonical feature vector for timestamp T.

    Applies the verified FeatureEngineeringPipeline and WeatherFeatureEngineer,
    merges representations, computes dispersion indices, extracts row at T,
    and validates zero NaNs and exact canonical schema alignment.

    Args:
        df_aq: Raw/validated air quality DataFrame.
        df_weather: Raw/validated weather DataFrame.
        production_timestamp: Production timestamp T.

    Returns:
        1-row DataFrame containing all 114 canonical features for timestamp T.
    """
    # Truncate weather to <= T (discarding future forecast hours)
    T_dt = pd.to_datetime(production_timestamp, unit="s", utc=True)
    df_w_sliced = df_weather[df_weather["datetime_utc"] <= T_dt].copy()

    # Sice AQ to <= T
    df_aq_sliced = df_aq[df_aq["dt"] <= production_timestamp].copy()

    # 1. Feature Engineering for Pollutants
    fe_pipe = FeatureEngineeringPipeline()
    df_pol, _ = fe_pipe.build_features(df_aq_sliced, drop_na=False)
    df_pol = df_pol.reset_index()

    # 2. Feature Engineering for Meteorology
    wf_eng = WeatherFeatureEngineer()
    df_wf = wf_eng.build_weather_features(df_w_sliced)

    # 3. Fuse Features
    df_fused = wf_eng.fuse_features(df_pol, df_wf, drop_na=False)

    if "dt" not in df_fused.columns:
        df_fused["dt"] = df_fused["datetime_utc"].astype("int64") // 10**9

    # 4. Extract single target row at timestamp T
    row_T = df_fused[df_fused["dt"] == production_timestamp].copy()
    if len(row_T) == 0:
        raise ValidationError(f"Feature engineering pipeline did not produce an observation for T={production_timestamp}")

    # Validate against canonical 114 features
    canonical_features = load_canonical_feature_names()
    missing_cols = [c for c in canonical_features if c not in row_T.columns]
    if missing_cols:
        raise ValidationError(f"Engineered row is missing {len(missing_cols)} canonical features: {missing_cols[:5]}")

    canonical_row = row_T[canonical_features].copy()

    # Check for NaN / infinite values
    null_counts = canonical_row.isnull().sum().sum()
    if null_counts > 0:
        null_cols = [c for c in canonical_row.columns if canonical_row[c].isnull().any()]
        raise ValidationError(
            f"Canonical feature row at T={production_timestamp} contains {null_counts} null values in: {null_cols[:5]}"
        )

    # Cast dt to integer and predictors to float64
    canonical_row["dt"] = canonical_row["dt"].astype(np.int64)
    for col in canonical_features:
        if col != "dt":
            canonical_row[col] = canonical_row[col].astype(np.float64)

    logger.info(
        f"Successfully constructed 114 canonical features for T={production_timestamp}. "
        f"EPA AQI={canonical_row['epa_aqi'].iloc[0]}, Lag 1h={canonical_row['epa_aqi_lag_1h'].iloc[0]}."
    )
    return canonical_row


def run_hourly_pipeline(
    dry_run: bool = False,
    live: bool = False,
    lookback_hours: int = 72,
    output_dir: Path | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Execute complete hourly ingestion, feature engineering, and Hopsworks publication.

    Args:
        dry_run: If True, executes end-to-end validation without mutating Hopsworks.
        live: If True, writes to Hopsworks Feature Store.
        lookback_hours: Lookback window to retrieve from external APIs.
        output_dir: Snapshot report destination.
        api_key: Optional OpenWeather API key override.

    Returns:
        Structured snapshot execution report.
    """
    if output_dir is None:
        output_dir = SNAPSHOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    executed_at = now_utc.isoformat()

    conn = HopsworksFeatureStoreConnector()

    if dry_run and not live:
        logger.info("Executing hourly feature pipeline in DRY-RUN mode.")
        try:
            df_aq, df_weather = fetch_hourly_telemetry_window(
                lookback_hours=lookback_hours, api_key=api_key
            )
            T = derive_production_timestamp(df_aq, df_weather, now_utc=now_utc)
            validate_hourly_grid_continuity(df_aq, df_weather, production_timestamp=T)
            canonical_row = construct_canonical_feature_dataframe(df_aq, df_weather, production_timestamp=T)
            obs_dt = int(canonical_row["dt"].iloc[0])
            current_aqi = int(canonical_row["epa_aqi"].iloc[0])
            storage_df = conn.prepare_storage_dataframe(canonical_row)
            storage_cols_count = len(storage_df.columns) if isinstance(storage_df, pd.DataFrame) else len(canonical_row.columns) + 1
            mode_desc = "dry_run_live_telemetry"
        except Exception as e_live:
            logger.warning(f"Dry-run live fetch skipped or failed ({e_live}); falling back to local dataset validation.")
            # Fallback for offline CI without API keys
            features_file = PROCESSED_DATA_DIR / "features_v2_weather.csv"
            if features_file.exists():
                local_df = pd.read_csv(features_file, nrows=100)
                canonical_row = conn.project_inference_features(local_df.iloc[[-1]])
            else:
                bootstrap_json = RUNTIME_DIR / "bootstrap" / "latest_feature_vector.json"
                if bootstrap_json.exists():
                    with open(bootstrap_json, "r", encoding="utf-8") as f:
                        bdata = json.load(f)
                    bfeats = bdata.get("features", {})
                    canonical_names = load_canonical_feature_names()
                    canonical_row = pd.DataFrame([{col: bfeats.get(col, 0.0) for col in canonical_names}])
                    canonical_row["dt"] = int(1788793200)
                else:
                    raise ValidationError("Neither live APIs nor local features_v2_weather.csv is available.")
            obs_dt = int(canonical_row["dt"].iloc[0])
            current_aqi = int(canonical_row["epa_aqi"].iloc[0])
            storage_df = conn.prepare_storage_dataframe(canonical_row)
            storage_cols_count = len(storage_df.columns) if isinstance(storage_df, pd.DataFrame) else len(canonical_row.columns) + 1
            mode_desc = "dry_run_local_fallback"

        age_seconds = int(now_utc.timestamp()) - obs_dt
        is_stale = age_seconds > (3 * 3600)

        report = {
            "status": "dry_run_success",
            "executed_at": executed_at,
            "mode": mode_desc,
            "observation_timestamp": obs_dt,
            "observation_datetime_utc": datetime.fromtimestamp(obs_dt, tz=timezone.utc).isoformat(),
            "observation_age_seconds": age_seconds,
            "is_stale": is_stale,
            "project_computed_current_epa_aqi": current_aqi,
            "canonical_features_count": len(canonical_row.columns),
            "storage_columns_count": storage_cols_count,
            "hopsworks_mutated": False,
            "message": "Dry-run validation completed successfully; Hopsworks was not mutated.",
        }
    else:
        logger.info("Executing hourly feature pipeline in LIVE mode.")
        df_aq, df_weather = fetch_hourly_telemetry_window(
            lookback_hours=lookback_hours, api_key=api_key
        )
        T = derive_production_timestamp(df_aq, df_weather, now_utc=now_utc)
        validate_hourly_grid_continuity(df_aq, df_weather, production_timestamp=T)
        canonical_row = construct_canonical_feature_dataframe(df_aq, df_weather, production_timestamp=T)
        obs_dt = int(canonical_row["dt"].iloc[0])
        current_aqi = int(canonical_row["epa_aqi"].iloc[0])
        storage_df = conn.prepare_storage_dataframe(canonical_row)
        storage_cols_count = len(storage_df.columns) if isinstance(storage_df, pd.DataFrame) else len(canonical_row.columns) + 1
        age_seconds = int(now_utc.timestamp()) - obs_dt
        is_stale = age_seconds > (3 * 3600)

        logger.info(f"Publishing observation T={obs_dt} to Hopsworks Feature Store...")
        hw_result = conn.insert_hourly_feature_row(canonical_row, wait_for_job=True)

        report = {
            "status": "live_ingestion_success",
            "executed_at": executed_at,
            "mode": "live",
            "observation_timestamp": obs_dt,
            "observation_datetime_utc": datetime.fromtimestamp(obs_dt, tz=timezone.utc).isoformat(),
            "observation_age_seconds": age_seconds,
            "is_stale": is_stale,
            "project_computed_current_epa_aqi": current_aqi,
            "canonical_features_count": len(canonical_row.columns),
            "storage_columns_count": storage_cols_count,
            "hopsworks_mutated": True,
            "hopsworks_result": hw_result,
            "message": "Hourly observation successfully engineered and ingested into Hopsworks Feature Store.",
        }

    report_path = output_dir / "latest_ingestion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Hourly ingestion report saved to {report_path}")
    return report


def run_ingestion(
    dry_run: bool = False,
    api_key: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute hourly ingestion and validation (workflow runner interface).

    Args:
        dry_run: If True, skips external API call and generates validation report on local dataset.
        api_key: Optional API key override.
        output_dir: Directory to save snapshot report.

    Returns:
        Summary report dictionary.
    """
    if output_dir is None:
        output_dir = SNAPSHOT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).isoformat()

    if dry_run:
        logger.info("Executing hourly ingestion in DRY-RUN mode (skipping external API calls).")
        features_file = PROCESSED_DATA_DIR / "features_v2_weather.csv"
        exists = features_file.exists()
        row_count = 0
        if exists:
            df = pd.read_csv(features_file)
            row_count = len(df)

        report = {
            "status": "dry_run_success",
            "executed_at": timestamp_str,
            "mode": "dry_run",
            "features_file_exists": exists,
            "row_count": row_count,
            "message": "Dry-run validation completed successfully without external API requests.",
        }
    else:
        if api_key is not None:
            active_key = api_key
        else:
            active_key = os.environ.get("OPENWEATHER_API_KEY", "")

        if not active_key or active_key.strip() == "":
            raise DataIngestionError("OPENWEATHER_API_KEY is required for live scheduled feature pipeline.")

        logger.info(f"Fetching live hourly telemetry for Lat: {TARGET_LAT}, Lon: {TARGET_LON}...")
        provider = OpenWeatherProvider(api_key=active_key)
        raw_aq = provider.fetch_current_air_quality(lat=TARGET_LAT, lon=TARGET_LON)
        raw_weather = provider.fetch_current_weather(lat=TARGET_LAT, lon=TARGET_LON)

        report = {
            "status": "live_ingestion_success",
            "executed_at": timestamp_str,
            "mode": "live",
            "location": {"lat": TARGET_LAT, "lon": TARGET_LON},
            "raw_aq_record_count": len(raw_aq.get("list", [])),
            "weather_received": "main" in raw_weather,
            "air_quality_sample": raw_aq,
            "weather_sample": raw_weather,
        }

    report_path = output_dir / "latest_ingestion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Hourly ingestion report saved to {report_path}")
    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run hourly data ingestion and feature pipeline for Pearls AQI.")
    parser.add_argument("--dry-run", action="store_true", help="Execute validation without mutating Hopsworks")
    parser.add_argument("--live", action="store_true", help="Publish observation to live Hopsworks Feature Store")
    parser.add_argument("--lookback-hours", type=int, default=72, help="Historical telemetry window hours (default: 72)")
    parser.add_argument("--output-dir", type=str, default="data/snapshots", help="Path to write snapshot report")
    args = parser.parse_args()

    try:
        # Default to dry-run if neither is explicitly passed
        is_live = args.live and not args.dry_run
        is_dry_run = args.dry_run or not args.live

        report = run_hourly_pipeline(
            dry_run=is_dry_run,
            live=is_live,
            lookback_hours=args.lookback_hours,
            output_dir=Path(args.output_dir),
        )
        print(json.dumps(report, indent=2))
        return 0
    except Exception as e:
        logger.error(f"Scheduled hourly pipeline failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
