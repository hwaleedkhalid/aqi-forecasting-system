"""Pearls AQI Predictor - Historical Data Backfill Pipeline.

Fetches and persists raw historical air pollution observations in monthly
chunks from OpenWeather API starting from November 27, 2020 to present.
Features resumable execution, atomic writes, and data completeness auditing.
"""

import argparse
import calendar
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from src.config import (
    API_RATE_LIMIT_DELAY,
    MIN_COMPLETENESS_RATIO,
    RAW_AQ_DIR,
    TARGET_LAT,
    TARGET_LON,
)
from src.data_ingestion.openweather_provider import (
    MIN_HISTORY_TIMESTAMP,
    OpenWeatherProvider,
)
from src.exceptions import DataIngestionError, ValidationError
from src.logger import logger


def generate_monthly_intervals(
    start_year: int = 2020,
    start_month: int = 11,
    end_dt: datetime | None = None,
) -> list[tuple[str, int, int]]:
    """Generate list of monthly (month_str, start_ts, end_ts) tuples in UTC.

    Args:
        start_year: Initial backfill year (default: 2020).
        start_month: Initial backfill month (default: 11 for Nov 2020).
        end_dt: End datetime in UTC. Defaults to current UTC time.

    Returns:
        List of tuples: (YYYY-MM, start_unix_ts, end_unix_ts).
    """
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)

    intervals: list[tuple[str, int, int]] = []
    current_year = start_year
    current_month = start_month

    while True:
        if (current_year > end_dt.year) or (
            current_year == end_dt.year and current_month > end_dt.month
        ):
            break

        month_label = f"{current_year:04d}-{current_month:02d}"
        _, last_day = calendar.monthrange(current_year, current_month)

        # November 2020 starts at Nov 27, 2020 13:00:00 UTC (MIN_HISTORY_TIMESTAMP)
        if current_year == 2020 and current_month == 11:
            month_start = datetime(2020, 11, 27, 13, 0, 0, tzinfo=timezone.utc)
        else:
            month_start = datetime(current_year, current_month, 1, 0, 0, 0, tzinfo=timezone.utc)

        # For current month, end at current hour boundary
        if current_year == end_dt.year and current_month == end_dt.month:
            month_end = end_dt.replace(minute=59, second=59, microsecond=0)
        else:
            month_end = datetime(current_year, current_month, last_day, 23, 59, 59, tzinfo=timezone.utc)

        start_ts = int(month_start.timestamp())
        end_ts = int(month_end.timestamp())

        if start_ts <= end_ts:
            intervals.append((month_label, start_ts, end_ts))

        # Advance to next month
        if current_month == 12:
            current_year += 1
            current_month = 1
        else:
            current_month += 1

    return intervals


class HistoricalBackfillService:
    """Service for managing historical air quality backfill and dataset auditing."""

    def __init__(
        self,
        provider: OpenWeatherProvider | None = None,
        output_dir: Path = RAW_AQ_DIR,
        target_lat: float = TARGET_LAT,
        target_lon: float = TARGET_LON,
    ) -> None:
        """Initialize HistoricalBackfillService.

        Args:
            provider: Data provider instance.
            output_dir: Directory where raw JSON chunks are stored.
            target_lat: Target latitude.
            target_lon: Target longitude.
        """
        self.provider = provider or OpenWeatherProvider()
        self.output_dir = output_dir
        self.target_lat = target_lat
        self.target_lon = target_lon

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def is_month_downloaded(self, month_label: str) -> bool:
        """Check if raw monthly file exists and contains valid records."""
        file_path = self.output_dir / f"{month_label}.json"
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return bool(isinstance(data, dict) and len(data.get("list", [])) > 0)
        except Exception:
            return False

    def save_raw_month(self, month_label: str, data: dict[str, Any]) -> Path:
        """Save raw API JSON response directly without modification.

        Args:
            month_label: Month identifier (e.g., '2021-05').
            data: Raw JSON payload from OpenWeather API.

        Returns:
            Path to the saved JSON file.
        """
        file_path = self.output_dir / f"{month_label}.json"
        temp_path = self.output_dir / f"{month_label}.json.tmp"

        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Atomic rename for duplicate-safe write
        temp_path.replace(file_path)
        return file_path

    def _fetch_and_save_single_month(
        self, month_label: str, start_ts: int, end_ts: int
    ) -> tuple[str, bool, int]:
        """Fetch and persist a single month's raw observations."""
        try:
            raw_payload = self.provider.fetch_historical_air_quality(
                lat=self.target_lat,
                lon=self.target_lon,
                start=start_ts,
                end=end_ts,
            )
            self.save_raw_month(month_label, raw_payload)
            records_count = len(raw_payload.get("list", []))
            logger.info(f"Saved {month_label}.json ({records_count} records)")
            return (month_label, True, records_count)
        except (DataIngestionError, ValidationError) as err:
            logger.error(f"Failed downloading {month_label}: {err}")
            return (month_label, False, 0)

    def run_backfill(
        self,
        force: bool = False,
        delay_seconds: float = API_RATE_LIMIT_DELAY,
        end_dt: datetime | None = None,
        max_workers: int = 5,
    ) -> dict[str, Any]:
        """Execute full backfill of historical air quality data.

        Args:
            force: If True, re-download all months even if already present.
            delay_seconds: Sleep interval between API calls for rate limiting.
            end_dt: End datetime cutoff.
            max_workers: Concurrent workers for downloading months.

        Returns:
            Audit dictionary containing data completeness statistics.
        """
        intervals = generate_monthly_intervals(end_dt=end_dt)
        logger.info(
            f"Starting historical backfill for coordinates ({self.target_lat}, {self.target_lon})"
        )
        logger.info(
            f"Total monthly intervals to process: {len(intervals)} (From {intervals[0][0]} to {intervals[-1][0]})"
        )

        tasks_to_run: list[tuple[str, int, int]] = []
        skipped_count = 0

        for month_label, start_ts, end_ts in intervals:
            if not force and self.is_month_downloaded(month_label):
                logger.debug(f"Skipping {month_label} (already downloaded)")
                skipped_count += 1
            else:
                tasks_to_run.append((month_label, start_ts, end_ts))

        logger.info(f"Queued {len(tasks_to_run)} months to download ({skipped_count} skipped).")

        downloaded_count = 0
        failed_count = 0

        if tasks_to_run:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(
                        self._fetch_and_save_single_month, label, start, end
                    ): label
                    for label, start, end in tasks_to_run
                }

                for future in as_completed(future_map):
                    label, success, _ = future.result()
                    if success:
                        downloaded_count += 1
                    else:
                        failed_count += 1

        logger.info(
            f"Backfill finished: {downloaded_count} downloaded, {skipped_count} skipped, {failed_count} failed."
        )

        # Audit dataset completeness
        audit_report = self.audit_dataset_completeness(intervals)
        self.save_audit_report(audit_report)
        return audit_report

    def audit_dataset_completeness(
        self, intervals: list[tuple[str, int, int]] | None = None
    ) -> dict[str, Any]:
        """Analyze completeness and missing intervals across all stored raw data.

        Args:
            intervals: Expected monthly intervals. If None, generated dynamically.

        Returns:
            Dictionary with completeness statistics and gap analysis.
        """
        if intervals is None:
            intervals = generate_monthly_intervals()

        if not intervals:
            return {"status": "error", "message": "No intervals specified"}

        overall_start_ts = intervals[0][1]
        overall_end_ts = intervals[-1][2]
        expected_total_hours = max(1, (overall_end_ts - overall_start_ts) // 3600 + 1)

        all_timestamps: set[int] = set()
        month_breakdown: list[dict[str, Any]] = []

        for month_label, start_ts, end_ts in intervals:
            file_path = self.output_dir / f"{month_label}.json"
            expected_month_hours = (end_ts - start_ts) // 3600 + 1

            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    records = data.get("list", [])
                    month_dts = {r["dt"] for r in records if "dt" in r}
                    all_timestamps.update(month_dts)
                    actual_count = len(month_dts)
                except Exception as err:
                    logger.warning(f"Error reading {file_path.name}: {err}")
                    actual_count = 0
            else:
                actual_count = 0

            month_ratio = (actual_count / expected_month_hours) if expected_month_hours > 0 else 0.0
            month_breakdown.append({
                "month": month_label,
                "expected_hours": expected_month_hours,
                "actual_records": actual_count,
                "completeness_ratio": round(month_ratio, 4),
            })

        actual_total_records = len(all_timestamps)
        completeness_ratio = round(actual_total_records / expected_total_hours, 4)

        # Gap detection
        sorted_dts = sorted(all_timestamps)
        gaps: list[dict[str, Any]] = []
        largest_gap_hours = 0

        for i in range(len(sorted_dts) - 1):
            diff_hours = (sorted_dts[i + 1] - sorted_dts[i]) // 3600
            if diff_hours > 1:
                gap_start_str = datetime.fromtimestamp(sorted_dts[i], tz=timezone.utc).isoformat()
                gap_end_str = datetime.fromtimestamp(sorted_dts[i + 1], tz=timezone.utc).isoformat()
                gap_len = diff_hours - 1
                gaps.append({
                    "from_utc": gap_start_str,
                    "to_utc": gap_end_str,
                    "missing_hours": gap_len,
                })
                if gap_len > largest_gap_hours:
                    largest_gap_hours = gap_len

        meets_target = completeness_ratio >= MIN_COMPLETENESS_RATIO

        report = {
            "target_city": "Lahore",
            "coordinates": {"lat": self.target_lat, "lon": self.target_lon},
            "date_range": {
                "start_utc": datetime.fromtimestamp(overall_start_ts, tz=timezone.utc).isoformat(),
                "end_utc": datetime.fromtimestamp(overall_end_ts, tz=timezone.utc).isoformat(),
            },
            "expected_hours": expected_total_hours,
            "actual_unique_records": actual_total_records,
            "completeness_ratio": completeness_ratio,
            "target_completeness_ratio": MIN_COMPLETENESS_RATIO,
            "meets_target_completeness": meets_target,
            "total_gaps_detected": len(gaps),
            "largest_gap_hours": largest_gap_hours,
            "gaps": gaps[:20],  # Sample up to top 20 gaps for summary
            "month_breakdown": month_breakdown,
        }

        logger.info(
            f"Dataset Completeness: {actual_total_records}/{expected_total_hours} "
            f"({completeness_ratio * 100:.2f}%) - Target >= {MIN_COMPLETENESS_RATIO * 100}%: {'PASS' if meets_target else 'FAIL'}"
        )

        return report

    def save_audit_report(self, report: dict[str, Any]) -> Path:
        """Save audit report to data/raw/air_quality/backfill_summary.json."""
        summary_path = self.output_dir / "backfill_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved dataset audit report to {summary_path}")
        return summary_path


def main() -> None:
    """CLI entry point for historical data backfill."""
    parser = argparse.ArgumentParser(description="Backfill historical air quality data.")
    parser.add_argument("--force", action="store_true", help="Force re-download all months.")
    parser.add_argument("--lat", type=float, default=TARGET_LAT, help="Target latitude.")
    parser.add_argument("--lon", type=float, default=TARGET_LON, help="Target longitude.")
    parser.add_argument(
        "--delay", type=float, default=API_RATE_LIMIT_DELAY, help="Delay between calls in seconds."
    )
    parser.add_argument(
        "--workers", type=int, default=5, help="Number of concurrent workers."
    )

    args = parser.parse_args()
    service = HistoricalBackfillService(target_lat=args.lat, target_lon=args.lon)
    service.run_backfill(force=args.force, delay_seconds=args.delay, max_workers=args.workers)


if __name__ == "__main__":
    main()
