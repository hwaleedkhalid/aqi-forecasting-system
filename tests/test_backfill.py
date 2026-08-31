"""Unit tests for Historical Data Backfill Pipeline.

Tests monthly interval generation, resumable fetching, atomic raw writes,
dataset completeness calculations, and gap auditing using mocked provider.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.data_ingestion.openweather_provider import OpenWeatherProvider
from src.exceptions import DataIngestionError
from src.feature_pipeline.backfill import (
    HistoricalBackfillService,
    generate_monthly_intervals,
)


@pytest.fixture
def mock_provider() -> MagicMock:
    """Create a mock OpenWeatherProvider."""
    provider = MagicMock(spec=OpenWeatherProvider)
    provider.fetch_historical_air_quality.return_value = {
        "coord": {"lon": 74.3436, "lat": 31.5497},
        "list": [
            {
                "dt": 1606482000,
                "main": {"aqi": 3},
                "components": {"co": 300, "no": 1, "no2": 20, "o3": 40, "so2": 10, "pm2_5": 30, "pm10": 60, "nh3": 5},
            },
            {
                "dt": 1606485600,
                "main": {"aqi": 3},
                "components": {"co": 300, "no": 1, "no2": 20, "o3": 40, "so2": 10, "pm2_5": 30, "pm10": 60, "nh3": 5},
            },
        ],
    }
    return provider


@pytest.fixture
def temp_raw_dir(tmp_path: Path) -> Path:
    """Create temporary directory for raw JSON output."""
    raw_dir = tmp_path / "raw_air_quality"
    raw_dir.mkdir()
    return raw_dir


# =============================================================================
# Interval Generation Tests
# =============================================================================

class TestIntervalGeneration:
    """Test monthly interval calculation and boundary logic."""

    def test_nov_2020_starts_at_earliest_boundary(self) -> None:
        end_dt = datetime(2020, 11, 30, 23, 59, 59, tzinfo=timezone.utc)
        intervals = generate_monthly_intervals(start_year=2020, start_month=11, end_dt=end_dt)
        assert len(intervals) == 1
        label, start_ts, end_ts = intervals[0]
        assert label == "2020-11"
        assert datetime.fromtimestamp(start_ts, tz=timezone.utc) == datetime(2020, 11, 27, 13, 0, 0, tzinfo=timezone.utc)

    def test_multiple_months_generation(self) -> None:
        end_dt = datetime(2021, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
        intervals = generate_monthly_intervals(start_year=2020, start_month=11, end_dt=end_dt)
        labels = [i[0] for i in intervals]
        assert labels == ["2020-11", "2020-12", "2021-01", "2021-02", "2021-03"]

    def test_leap_year_february_days(self) -> None:
        end_dt = datetime(2024, 2, 29, 23, 59, 59, tzinfo=timezone.utc)
        intervals = generate_monthly_intervals(start_year=2024, start_month=2, end_dt=end_dt)
        assert len(intervals) == 1
        _, start_ts, end_ts = intervals[0]
        # Feb 2024 has 29 days = 696 hours
        hours = (end_ts - start_ts) // 3600 + 1
        assert hours == 29 * 24


# =============================================================================
# Backfill Service Tests
# =============================================================================

class TestBackfillService:
    """Test backfill execution, persistence, and resumability."""

    def test_is_month_downloaded_nonexistent(self, temp_raw_dir: Path) -> None:
        service = HistoricalBackfillService(output_dir=temp_raw_dir)
        assert service.is_month_downloaded("2021-01") is False

    def test_is_month_downloaded_valid(self, temp_raw_dir: Path) -> None:
        service = HistoricalBackfillService(output_dir=temp_raw_dir)
        sample_file = temp_raw_dir / "2021-01.json"
        with open(sample_file, "w", encoding="utf-8") as f:
            json.dump({"list": [{"dt": 1609459200}]}, f)

        assert service.is_month_downloaded("2021-01") is True

    def test_save_raw_month_atomic(self, temp_raw_dir: Path) -> None:
        service = HistoricalBackfillService(output_dir=temp_raw_dir)
        payload = {"coord": {"lat": 31.5, "lon": 74.3}, "list": [{"dt": 100}]}
        saved_path = service.save_raw_month("2021-05", payload)

        assert saved_path.exists()
        with open(saved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == payload

    def test_run_backfill_downloads_and_resumes(
        self, temp_raw_dir: Path, mock_provider: MagicMock
    ) -> None:
        service = HistoricalBackfillService(provider=mock_provider, output_dir=temp_raw_dir)
        end_dt = datetime(2020, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

        # First run (2 months: Nov 2020, Dec 2020)
        report = service.run_backfill(force=False, delay_seconds=0, end_dt=end_dt)
        assert mock_provider.fetch_historical_air_quality.call_count == 2
        assert report["meets_target_completeness"] is not None

        # Second run without force should skip both
        mock_provider.reset_mock()
        service.run_backfill(force=False, delay_seconds=0, end_dt=end_dt)
        assert mock_provider.fetch_historical_air_quality.call_count == 0

        # Run with force should re-download
        service.run_backfill(force=True, delay_seconds=0, end_dt=end_dt)
        assert mock_provider.fetch_historical_air_quality.call_count == 2

    def test_run_backfill_handles_provider_error(
        self, temp_raw_dir: Path, mock_provider: MagicMock
    ) -> None:
        mock_provider.fetch_historical_air_quality.side_effect = DataIngestionError("Network down")
        service = HistoricalBackfillService(provider=mock_provider, output_dir=temp_raw_dir)
        end_dt = datetime(2020, 11, 30, 23, 59, 59, tzinfo=timezone.utc)

        # Should log error and continue without crashing
        report = service.run_backfill(force=True, delay_seconds=0, end_dt=end_dt)
        assert report["actual_unique_records"] == 0


# =============================================================================
# Completeness & Auditing Tests
# =============================================================================

class TestCompletenessAuditing:
    """Test auditing logic, gap detection, and completeness percentage."""

    def test_audit_dataset_completeness_and_gap_detection(
        self, temp_raw_dir: Path
    ) -> None:
        service = HistoricalBackfillService(output_dir=temp_raw_dir)
        intervals = [("2021-01", 1609459200, 1609459200 + 3600 * 9)]  # 10 hours expected

        # Simulate 8 hours recorded with a 2-hour gap in between
        dts = [
            1609459200,
            1609459200 + 3600,
            1609459200 + 7200,
            # Gap of 2 hours
            1609459200 + 18000,
            1609459200 + 21600,
        ]
        with open(temp_raw_dir / "2021-01.json", "w", encoding="utf-8") as f:
            json.dump({"list": [{"dt": t} for t in dts]}, f)

        report = service.audit_dataset_completeness(intervals)
        assert report["expected_hours"] == 10
        assert report["actual_unique_records"] == 5
        assert report["completeness_ratio"] == 0.5
        assert report["total_gaps_detected"] == 1
        assert report["largest_gap_hours"] == 2
        assert report["meets_target_completeness"] is False
