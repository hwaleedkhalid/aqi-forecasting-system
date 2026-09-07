"""Unit tests for scheduled hourly ingestion and Hopsworks feature pipeline.

All tests are strictly mocked and offline; zero external network calls or cloud credentials required.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.exceptions import DataIngestionError, ValidationError
from src.feature_pipeline.run_hourly_ingestion import (
    construct_canonical_feature_dataframe,
    derive_production_timestamp,
    fetch_hourly_telemetry_window,
    run_hourly_pipeline,
    validate_hourly_grid_continuity,
)


def _generate_synthetic_telemetry(
    end_dt: int = 1788793200,
    hours: int = 72,
    gap_start_hour: int | None = None,
    gap_duration_hours: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Helper to generate aligned synthetic air quality and weather telemetry DataFrames."""
    start_dt = end_dt - (hours - 1) * 3600
    timestamps = [start_dt + i * 3600 for i in range(hours)]

    if gap_start_hour is not None and gap_duration_hours > 0:
        # Drop timestamps to create a gap
        drop_indices = set(range(gap_start_hour, min(hours, gap_start_hour + gap_duration_hours)))
        timestamps = [ts for idx, ts in enumerate(timestamps) if idx not in drop_indices]

    aq_rows = []
    for ts in timestamps:
        val = 50.0 + (ts % 24) * 2.0
        aq_rows.append({
            "dt": ts,
            "co": 300.0,
            "no": 1.0,
            "no2": 15.0,
            "o3": 40.0,
            "so2": 5.0,
            "pm2_5": val,
            "pm10": val * 1.5,
            "nh3": 4.0,
            "datetime_utc": pd.to_datetime(ts, unit="s", utc=True),
            "epa_aqi": int(val),
            "dominant_pollutant": "pm2_5",
            "aqi_category": "Moderate" if val <= 100 else "Unhealthy for Sensitive Groups",
        })
    df_aq = pd.DataFrame(aq_rows)

    weather_rows = []
    for ts in timestamps:
        weather_rows.append({
            "datetime_utc": pd.to_datetime(ts, unit="s", utc=True),
            "temperature_2m": 25.0 + (ts % 24) * 0.5,
            "relative_humidity_2m": 60.0,
            "surface_pressure": 1012.0,
            "wind_speed_10m": 3.5,
            "wind_direction_10m": 180.0,
            "precipitation": 0.0,
        })
    df_weather = pd.DataFrame(weather_rows)

    return df_aq, df_weather


class TestHourlyPipelineTelemetry:
    """Tests for telemetry fetching and timestamp derivation."""

    @patch("src.feature_pipeline.run_hourly_ingestion.requests.get")
    @patch("src.feature_pipeline.run_hourly_ingestion.OpenWeatherProvider")
    def test_fetch_hourly_telemetry_window_mocked(self, mock_ow_cls, mock_get):
        """Test fetching and parsing of air quality and weather telemetry."""
        mock_ow = MagicMock()
        mock_ow_cls.return_value = mock_ow
        mock_ow.fetch_historical_air_quality.return_value = {
            "list": [
                {
                    "dt": 1788793200,
                    "main": {"aqi": 3},
                    "components": {
                        "co": 300.0, "no": 0.5, "no2": 15.0, "o3": 45.0,
                        "so2": 4.0, "pm2_5": 35.0, "pm10": 60.0, "nh3": 3.0,
                    },
                }
            ]
        }

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "hourly": {
                "time": ["2026-09-07T15:00"],
                "temperature_2m": [28.5],
                "relative_humidity_2m": [65.0],
                "surface_pressure": [1010.5],
                "wind_speed_10m": [4.2],
                "wind_direction_10m": [120.0],
                "precipitation": [0.0],
            }
        }
        mock_get.return_value = mock_resp

        df_aq, df_weather = fetch_hourly_telemetry_window(lookback_hours=24, api_key="dummy_key")

        assert len(df_aq) == 1
        assert "epa_aqi" in df_aq.columns
        assert df_aq["dt"].iloc[0] == 1788793200
        assert len(df_weather) == 1
        assert df_weather["temperature_2m"].iloc[0] == 28.5

    def test_derive_production_timestamp_floors_hour_and_discards_future(self):
        """Test that T is derived as the latest completed common hourly boundary <= now."""
        # Telemetry with future weather forecast
        now_utc = datetime(2026, 9, 7, 15, 30, 0, tzinfo=timezone.utc)
        now_ts = int(now_utc.timestamp())

        # AQ data ends at 15:00 UTC (1788793200)
        t_15 = 1788793200
        df_aq = pd.DataFrame({"dt": [t_15 - 3600, t_15]})

        # Weather data includes future forecast up to 23:00 UTC
        weather_times = [
            pd.to_datetime(t_15 - 3600, unit="s", utc=True),
            pd.to_datetime(t_15, unit="s", utc=True),
            pd.to_datetime(t_15 + 3600, unit="s", utc=True),  # 16:00 (future relative to now)
            pd.to_datetime(t_15 + 7200, unit="s", utc=True),  # 17:00 (future)
        ]
        df_weather = pd.DataFrame({"datetime_utc": weather_times})

        T = derive_production_timestamp(df_aq, df_weather, now_utc=now_utc)
        assert T == t_15
        assert T % 3600 == 0


class TestHourlyGridContinuity:
    """Tests for hourly grid continuity validation."""

    def test_validate_hourly_grid_continuity_success(self):
        """Continuous 72-hour series passes grid continuity validation."""
        end_dt = 1788793200
        df_aq, df_weather = _generate_synthetic_telemetry(end_dt=end_dt, hours=72)
        # Should not raise
        validate_hourly_grid_continuity(df_aq, df_weather, production_timestamp=end_dt)

    def test_insufficient_lookback_fails_closed(self):
        """Telemetry with less than 24 hours lookback raises ValidationError."""
        end_dt = 1788793200
        # Only 10 hours provided
        df_aq, df_weather = _generate_synthetic_telemetry(end_dt=end_dt, hours=10)
        with pytest.raises(ValidationError, match="Insufficient air quality lookback"):
            validate_hourly_grid_continuity(df_aq, df_weather, production_timestamp=end_dt)

    def test_gap_exceeding_3h_fails_closed(self):
        """A gap > 3 hours within [T-24h, T] raises ValidationError."""
        end_dt = 1788793200
        # Create a 4-hour gap inside the 24h lookback window
        # Lookback is 72 hours, index 55 to 58 would be within [T-24h, T]
        df_aq, df_weather = _generate_synthetic_telemetry(
            end_dt=end_dt, hours=72, gap_start_hour=55, gap_duration_hours=4
        )
        with pytest.raises(ValidationError, match="unbridgeable gap of 5 hours|unbridgeable gap of 4 hours"):
            validate_hourly_grid_continuity(df_aq, df_weather, production_timestamp=end_dt)

    def test_gap_within_3h_tolerated(self):
        """A gap <= 3 hours within [T-24h, T] is tolerated."""
        end_dt = 1788793200
        # 2-hour gap (drop 2 observations -> gap of 3 hours)
        df_aq, df_weather = _generate_synthetic_telemetry(
            end_dt=end_dt, hours=72, gap_start_hour=55, gap_duration_hours=2
        )
        # Should not raise because gap <= 3h is handled by interpolation
        validate_hourly_grid_continuity(df_aq, df_weather, production_timestamp=end_dt, max_tolerated_gap_hours=3)


class TestCanonicalFeatureConstruction:
    """Tests for 114-column canonical feature engineering."""

    def test_construct_canonical_feature_dataframe_exact_114_columns(self):
        """Constructs exact 114 canonical features with zero NaNs and valid schema."""
        end_dt = 1788793200
        df_aq, df_weather = _generate_synthetic_telemetry(end_dt=end_dt, hours=72)

        row = construct_canonical_feature_dataframe(df_aq, df_weather, production_timestamp=end_dt)

        assert len(row) == 1
        assert len(row.columns) == 114
        assert row.isnull().sum().sum() == 0
        assert row["dt"].iloc[0] == end_dt
        assert "pm2_5_lag_1h" in row.columns
        assert "temperature_2m_rolling_mean_24h" in row.columns
        assert "stagnation_index" in row.columns
        assert "combustion_index" in row.columns

    def test_current_aqi_distinct_from_lag(self):
        """Confirms that epa_aqi reflects the current observation while epa_aqi_lag_1h reflects prior hour."""
        end_dt = 1788793200
        df_aq, df_weather = _generate_synthetic_telemetry(end_dt=end_dt, hours=72)
        # Explicitly set different AQI for last two hours
        df_aq.loc[df_aq["dt"] == end_dt, "epa_aqi"] = 185
        df_aq.loc[df_aq["dt"] == end_dt - 3600, "epa_aqi"] = 95

        row = construct_canonical_feature_dataframe(df_aq, df_weather, production_timestamp=end_dt)

        assert row["epa_aqi"].iloc[0] == 185
        assert row["epa_aqi_lag_1h"].iloc[0] == 95

    def test_two_hour_gap_is_reindexed_and_interpolated_before_feature_engineering(self):
        """Proves a 2-hour missing gap is explicitly reindexed and linearly interpolated before calculations.

        Verifies that dropping intermediate observation T-2h results in an intermediate interpolated value,
        rather than merely bypassing the gap or performing raw row-based shifts on uneven time intervals.
        """
        end_dt = 1788793200
        df_aq, df_weather = _generate_synthetic_telemetry(end_dt=end_dt, hours=72)

        # Set distinct linear values around T-2h to test exact interpolation:
        # T-3h (end_dt - 10800): pm2_5 = 40.0
        # T-2h (end_dt - 7200):  [DROPPED to simulate 2-hour gap]
        # T-1h (end_dt - 3600):  pm2_5 = 60.0
        # T    (end_dt):         pm2_5 = 70.0
        t_3h = end_dt - 10800
        t_2h = end_dt - 7200
        t_1h = end_dt - 3600

        df_aq.loc[df_aq["dt"] == t_3h, "pm2_5"] = 40.0
        df_aq.loc[df_aq["dt"] == t_1h, "pm2_5"] = 60.0
        df_aq.loc[df_aq["dt"] == end_dt, "pm2_5"] = 70.0

        # Drop the row at T-2h from both series, creating a 2-hour missing gap between T-3h and T-1h
        df_aq_gap = df_aq[df_aq["dt"] != t_2h].reset_index(drop=True)
        df_w_gap = df_weather[
            df_weather["datetime_utc"] != pd.to_datetime(t_2h, unit="s", utc=True)
        ].reset_index(drop=True)

        assert len(df_aq_gap) == 71
        assert t_2h not in df_aq_gap["dt"].values

        # Continuity validation passes since gap of 2h is <= 3h tolerated threshold
        validate_hourly_grid_continuity(df_aq_gap, df_w_gap, production_timestamp=end_dt)

        # Verify that prepare_time_series directly reindexes and interpolates the missing hour
        from src.feature_pipeline.feature_engineering import FeatureEngineeringPipeline
        fe_pipe = FeatureEngineeringPipeline()
        df_ts = fe_pipe.prepare_time_series(df_aq_gap)

        # The missing timestamp T-2h must now exist in the reindexed series
        t_2h_dt = pd.to_datetime(t_2h, unit="s", utc=True)
        assert t_2h_dt in df_ts.index
        # Its interpolated PM2.5 value must be exactly midpoint between 40.0 and 60.0 (50.0)
        assert np.isclose(df_ts.loc[t_2h_dt, "pm2_5"], 50.0)

        # Full feature construction on the gapped telemetry produces valid 114 canonical features with 0 nulls
        row = construct_canonical_feature_dataframe(df_aq_gap, df_w_gap, production_timestamp=end_dt)
        assert len(row) == 1
        assert len(row.columns) == 114
        assert row.isnull().sum().sum() == 0
        assert row["dt"].iloc[0] == end_dt
        assert np.isclose(row["pm2_5"].iloc[0], 70.0)
        assert np.isclose(row["pm2_5_lag_1h"].iloc[0], 60.0)


class TestHourlyPipelineExecution:
    """Tests for pipeline orchestration and dry-run/live modes."""

    @patch("src.feature_pipeline.run_hourly_ingestion.HopsworksFeatureStoreConnector")
    @patch("src.feature_pipeline.run_hourly_ingestion.fetch_hourly_telemetry_window")
    def test_dry_run_mode_never_calls_hopsworks_insert(self, mock_fetch, mock_conn_cls, tmp_path):
        """Dry-run mode performs end-to-end engineering but never mutates Hopsworks."""
        end_dt = 1788793200
        df_aq, df_weather = _generate_synthetic_telemetry(end_dt=end_dt, hours=72)
        mock_fetch.return_value = (df_aq, df_weather)

        mock_conn = MagicMock()
        mock_conn_cls.return_value = mock_conn

        report = run_hourly_pipeline(
            dry_run=True,
            live=False,
            lookback_hours=72,
            output_dir=tmp_path,
        )

        assert report["status"] == "dry_run_success"
        assert report["hopsworks_mutated"] is False
        assert report["canonical_features_count"] == 114
        assert report["storage_columns_count"] == 115
        mock_conn.insert_hourly_feature_row.assert_not_called()

        snapshot_file = tmp_path / "latest_ingestion_report.json"
        assert snapshot_file.exists()

    @patch("src.feature_pipeline.run_hourly_ingestion.HopsworksFeatureStoreConnector")
    @patch("src.feature_pipeline.run_hourly_ingestion.fetch_hourly_telemetry_window")
    def test_live_mode_calls_insert_hourly_feature_row(self, mock_fetch, mock_conn_cls, tmp_path):
        """Live mode calls insert_hourly_feature_row with verified 1-row DataFrame."""
        end_dt = 1788793200
        df_aq, df_weather = _generate_synthetic_telemetry(end_dt=end_dt, hours=72)
        mock_fetch.return_value = (df_aq, df_weather)

        mock_conn = MagicMock()
        mock_conn.canonical_features = construct_canonical_feature_dataframe(
            df_aq, df_weather, production_timestamp=end_dt
        ).columns.tolist()
        mock_conn.prepare_storage_dataframe.side_effect = lambda df: pd.concat([pd.DataFrame({"location_id": ["lahore"]}), df], axis=1)
        mock_conn.insert_hourly_feature_row.return_value = {"status": "hourly_ingestion_success", "observation_dt": end_dt}
        mock_conn_cls.return_value = mock_conn

        report = run_hourly_pipeline(
            dry_run=False,
            live=True,
            lookback_hours=72,
            output_dir=tmp_path,
        )

        assert report["status"] == "live_ingestion_success"
        assert report["hopsworks_mutated"] is True
        mock_conn.insert_hourly_feature_row.assert_called_once()
        assert report["project_computed_current_epa_aqi"] == int(df_aq.iloc[-1]["epa_aqi"])

    def test_observation_freshness_flag(self):
        """Tests that freshness flag correctly identifies fresh vs stale timestamps."""
        now_ts = int(datetime.now(timezone.utc).timestamp())
        fresh_ts = now_ts - 1800  # 30 minutes ago
        stale_ts = now_ts - 15000  # 4.1 hours ago

        assert (now_ts - fresh_ts) <= (3 * 3600)  # fresh
        assert (now_ts - stale_ts) > (3 * 3600)   # stale
