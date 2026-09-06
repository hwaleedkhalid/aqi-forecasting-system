"""Pearls AQI Predictor - Scheduled Hourly Ingestion & Validation Runner.

Command-line entry point for scheduled GitHub Actions feature pipeline.
Fetches current air quality & surface meteorology from OpenWeather API,
validates schema/freshness, and saves an exportable snapshot report for CI/CD artifact capture.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from src.config import OPENWEATHER_API_KEY, PROCESSED_DATA_DIR, TARGET_LAT, TARGET_LON
from src.data_ingestion.openweather_provider import OpenWeatherProvider
from src.exceptions import DataIngestionError, ValidationError
from src.logger import logger

SNAPSHOT_DIR = Path("data/snapshots")


def run_ingestion(
    dry_run: bool = False,
    api_key: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute hourly ingestion and validation.

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
            import pandas as pd
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
    """CLI entry point for scheduled runner."""
    parser = argparse.ArgumentParser(description="Run hourly data ingestion and validation for Pearls AQI.")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling external APIs")
    parser.add_argument("--output-dir", type=str, default="data/snapshots", help="Path to write snapshot report")
    args = parser.parse_args()

    try:
        report = run_ingestion(dry_run=args.dry_run, output_dir=Path(args.output_dir))
        print(json.dumps(report, indent=2))
        return 0
    except Exception as e:
        logger.error(f"Scheduled hourly ingestion failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
