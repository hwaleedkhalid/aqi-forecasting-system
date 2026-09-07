"""Pearls AQI Predictor - Historical Feature Store Backfill Pipeline.

Loads validated historical processed observations (48,715 hourly observations spanning
November 2020 through August 2026 for Lahore), verifies the canonical 114-column contract,
and uploads records to the live Hopsworks Feature Store ('aqi_weather_features_v2').

Architecture:
- primary_key=["location_id"], event_time="dt", online_enabled=True.
- Offline store: Historical dataset of 48,715 rows ingested with storage="offline".
- Online store: Single latest observation (dt = 1788159600) synchronized with upsert_if_newer=True.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_FEATURE_GROUP_NAME,
    HOPSWORKS_FEATURE_GROUP_VERSION,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT,
    PROCESSED_DATA_DIR,
)
from src.exceptions import FeatureStoreError, ValidationError
from src.feature_pipeline.hopsworks_integration import (
    HopsworksFeatureStoreConnector,
    load_canonical_feature_names,
)
from src.logger import logger

CSV_PATH = PROCESSED_DATA_DIR / "features_v2_weather.csv"
SCHEMA_PATH = PROCESSED_DATA_DIR / "feature_schema_v2_weather.json"


def audit_historical_dataset(df: pd.DataFrame, canonical_features: list[str]) -> dict[str, Any]:
    """Audit the historical dataset against the canonical feature schema and data quality rules.

    Args:
        df: Input raw processed DataFrame.
        canonical_features: List of 114 canonical feature names.

    Returns:
        Audit report dictionary.

    Raises:
        ValidationError: If dataset fails integrity, dimension, or quality checks.
    """
    row_count = len(df)
    if row_count == 0:
        raise ValidationError("Historical dataset is empty.")

    # 1. Verify all 114 canonical features exist
    missing_features = [f for f in canonical_features if f not in df.columns]
    if missing_features:
        raise ValidationError(
            f"Dataset missing {len(missing_features)} canonical features: {missing_features[:5]}"
        )

    # 2. Check event time dt
    if "dt" not in df.columns:
        raise ValidationError("Dataset is missing event_time column 'dt'")

    # Validate dt values are integral
    dt_numeric = pd.to_numeric(df["dt"], errors="coerce")
    if dt_numeric.isna().any():
        raise ValidationError("Non-numeric or NaN values detected in event_time column 'dt'")
    if not np.all(dt_numeric % 1 == 0):
        raise ValidationError("Fractional (non-integral) timestamps detected in event_time column 'dt'")

    dt_series = dt_numeric.astype(np.int64)
    min_dt = int(dt_series.min())
    max_dt = int(dt_series.max())

    # 3. Check duplicate timestamps
    duplicate_count = int(dt_series.duplicated().sum())
    if duplicate_count > 0:
        raise ValidationError(f"Dataset contains {duplicate_count} duplicate timestamp records.")

    # 4. Check NaN and Inf across canonical features
    feature_df = df[canonical_features]
    nan_count = int(feature_df.isna().sum().sum())
    if nan_count > 0:
        raise ValidationError(f"Dataset contains {nan_count} NaN values in canonical features.")

    inf_count = 0
    for col in canonical_features:
        if np.issubdtype(feature_df[col].dtype, np.number):
            inf_count += int(np.isinf(feature_df[col]).sum())
    if inf_count > 0:
        raise ValidationError(f"Dataset contains {inf_count} infinite values in canonical features.")

    # 5. Timestamp span and gap analysis
    total_span_hours = (max_dt - min_dt) // 3600 + 1
    gap_hours = total_span_hours - row_count
    completeness_pct = (row_count / total_span_hours) * 100.0

    earliest_iso = str(pd.to_datetime(min_dt, unit="s", utc=True))
    latest_iso = str(pd.to_datetime(max_dt, unit="s", utc=True))

    return {
        "row_count": row_count,
        "column_count": len(df.columns),
        "canonical_feature_count": len(canonical_features),
        "earliest_dt": min_dt,
        "latest_dt": max_dt,
        "earliest_iso": earliest_iso,
        "latest_iso": latest_iso,
        "total_calendar_span_hours": total_span_hours,
        "missing_gap_hours": gap_hours,
        "completeness_percentage": round(completeness_pct, 2),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_count": duplicate_count,
    }


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    """Partition a DataFrame into chunks of specified maximum size."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    return [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]


def wait_for_materialization_idle(fg: Any, timeout_seconds: int = 600, poll_interval: int = 10) -> str:
    """Wait until any active Spark materialization job on the Feature Group completes."""
    job = getattr(fg, "materialization_job", None)
    if job is None:
        return "NO_JOB"

    start_wait = time.time()
    while time.time() - start_wait < timeout_seconds:
        try:
            executions = job.get_executions()
            if not executions:
                return "NO_EXECUTIONS"
            latest = executions[0]
            state = getattr(latest, "state", "UNKNOWN")
            final_status = getattr(latest, "final_status", "UNDEFINED")
            logger.info(
                f"Materialization execution {latest.id}: state={state}, final_status={final_status}"
            )
            if state in ["FINISHED", "FAILED", "KILLED"]:
                if final_status in ["FAILED", "KILLED"]:
                    raise FeatureStoreError(
                        f"Materialization execution {latest.id} failed with status {final_status}"
                    )
                return str(final_status)
        except FeatureStoreError:
            raise
        except Exception as e:
            logger.warning(f"Querying materialization status: {e}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Materialization job did not finish within {timeout_seconds}s.")


class HopsworksBackfillRunner:
    """Orchestrates validation, chunking, and live ingestion into Hopsworks Feature Store."""

    def __init__(
        self,
        connector: HopsworksFeatureStoreConnector | None = None,
        csv_path: Path = CSV_PATH,
        chunk_size: int = 50000,
        wait: bool = True,
        storage: str = "offline",
    ) -> None:
        """Initialize Hopsworks backfill runner.

        Args:
            connector: Configured HopsworksFeatureStoreConnector instance.
            csv_path: Path to validated historical features CSV.
            chunk_size: Maximum batch size per upload chunk (default 50,000 for single batch).
            wait: If True, waits synchronously for ingestion completion.
            storage: Storage mode ('offline' [default], 'online', or 'all').
        """
        self.connector = connector or HopsworksFeatureStoreConnector()
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.wait = wait
        self.storage = storage
        self.canonical_features = load_canonical_feature_names()

    def run(
        self,
        dry_run: bool = False,
        verify_after: bool = True,
        sync_online: bool = True,
    ) -> dict[str, Any]:
        """Execute the backfill workflow.

        Args:
            dry_run: If True, executes audit and chunk planning without connecting to cloud.
            verify_after: If True, queries Hopsworks after ingestion to verify storage and parity.
            sync_online: If True and storage is offline, syncs latest observation to online store.

        Returns:
            Execution summary dictionary.
        """
        logger.info(f"Loading historical features from {self.csv_path}...")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Historical feature CSV not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Pre-flight audit
        audit = audit_historical_dataset(df, self.canonical_features)
        logger.info(
            f"Pre-flight audit passed: {audit['row_count']} hourly observations spanning "
            f"{audit['earliest_iso']} through {audit['latest_iso']} "
            f"({audit['missing_gap_hours']} hourly timestamps are absent across the covered calendar span, "
            f"giving {audit['completeness_percentage']}% temporal coverage)."
        )

        chunks = chunk_dataframe(df, self.chunk_size)
        logger.info(
            f"Planned {len(chunks)} chunks of max size {self.chunk_size} "
            f"(storage mode: {self.storage})."
        )

        if dry_run:
            logger.info("DRY-RUN mode active. Validating storage preparation without cloud writes...")
            storage_sample = self.connector.prepare_storage_dataframe(chunks[0].iloc[:10])
            if len(storage_sample.columns) != 115:
                raise ValidationError(
                    f"Expected 115 storage columns, got {len(storage_sample.columns)}"
                )
            if "location_id" not in storage_sample.columns or "dt" not in storage_sample.columns:
                raise ValidationError("Prepared storage dataframe missing location_id or dt")
            if storage_sample["dt"].dtype != np.int64:
                raise ValidationError(f"Expected dt to be int64, got {storage_sample['dt'].dtype}")

            # Validate inference projection
            projected = self.connector.project_inference_features(storage_sample)
            if len(projected.columns) != 114:
                raise ValidationError(
                    f"Expected 114 projected inference columns, got {len(projected.columns)}"
                )

            return {
                "status": "dry_run_success",
                "audit": audit,
                "chunk_count": len(chunks),
                "chunk_size": self.chunk_size,
                "storage_columns": len(storage_sample.columns),
                "projected_features": len(projected.columns),
                "storage_mode": self.storage,
            }

        # Live execution
        if not self.connector.is_cloud_configured():
            raise FeatureStoreError(
                "Hopsworks cloud credentials not configured. Provide HOPSWORKS_API_KEY."
            )

        logger.info(
            f"Connecting to Hopsworks project '{self.connector.project_name}' on {self.connector.host}..."
        )
        fg = self.connector.get_or_create_feature_group()
        logger.info(
            f"Feature Group '{fg.name}' v{fg.version} active "
            f"(PK={fg.primary_key}, event_time={fg.event_time}, online={fg.online_enabled})."
        )

        # Check if previous job execution is still running before starting
        logger.info("Verifying cluster job status...")
        wait_for_materialization_idle(fg, timeout_seconds=300, poll_interval=10)

        total_inserted = 0
        start_time = time.time()

        storage_param = None if self.storage == "all" else self.storage

        # 1. Execute historical backfill
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(
                f"Uploading chunk {idx}/{len(chunks)} ({len(chunk)} rows, "
                f"dt: {int(chunk['dt'].min())} -> {int(chunk['dt'].max())}) with storage='{self.storage}'..."
            )
            res = self.connector.insert_features(
                chunk,
                wait=False,  # Start ingestion and monitor via cluster execution API
                storage=storage_param,
            )
            total_inserted += len(chunk)
            logger.info(
                f"Chunk {idx}/{len(chunks)} uploaded to Hopsworks dataset storage. Job: {res.get('job_result')}"
            )

            if self.wait:
                logger.info(f"Waiting for Spark materialization job to complete chunk {idx}...")
                status = wait_for_materialization_idle(fg, timeout_seconds=900, poll_interval=15)
                logger.info(f"Chunk {idx} materialized successfully (status: {status}).")

        duration_sec = round(time.time() - start_time, 2)
        logger.info(f"Historical upload finished in {duration_sec}s ({total_inserted} rows).")

        # 2. Synchronize latest observation to online store (if storage is offline)
        online_sync_result = None
        if sync_online and self.storage == "offline":
            latest_row = df.sort_values(by="dt", ascending=False).iloc[[0]]
            latest_dt = int(latest_row["dt"].iloc[0])
            logger.info(
                f"Synchronizing latest observation (dt={latest_dt}, 2026-08-31 07:00:00 UTC) "
                f"to online store with upsert_if_newer=True..."
            )
            online_res = self.connector.insert_features(
                latest_row,
                wait=True,
                storage="online",
                write_options={"online_ingestion_options": {"upsert_if_newer": True}},
            )
            online_sync_result = {
                "latest_dt": latest_dt,
                "job_result": online_res.get("job_result"),
                "status": "synchronized",
            }
            logger.info(f"Online store synchronization completed: {online_sync_result}")

        result: dict[str, Any] = {
            "status": "completed",
            "feature_group": self.connector.feature_group_name,
            "version": self.connector.feature_group_version,
            "total_rows_ingested": total_inserted,
            "chunks_uploaded": len(chunks),
            "storage_mode": self.storage,
            "duration_seconds": duration_sec,
            "audit": audit,
            "online_sync": online_sync_result,
        }

        if verify_after:
            logger.info("Executing post-backfill remote verification...")
            verification = self.verify_remote_state(df)
            result["verification"] = verification

        return result

    def verify_remote_state(self, local_df: pd.DataFrame) -> dict[str, Any]:
        """Query remote Hopsworks Feature Store to verify offline history, online record, and parity."""
        fs = self.connector._login()
        fg = fs.get_feature_group(
            name=self.connector.feature_group_name,
            version=self.connector.feature_group_version,
        )

        # 1. Query remote offline data for lahore
        logger.info("Reading offline store data for Lahore via Feature Query Service...")
        query = fg.select_all().filter(fg.location_id == self.connector.location_id)
        remote_df = query.read()

        if remote_df is None or len(remote_df) == 0:
            raise FeatureStoreError("Remote feature group returned empty dataset after ingestion.")

        remote_rows = len(remote_df)
        remote_columns = len(remote_df.columns)
        earliest_remote_dt = int(remote_df["dt"].min())
        latest_remote_dt = int(remote_df["dt"].max())

        local_latest = local_df.sort_values(by="dt", ascending=False).iloc[0]
        expected_latest_dt = int(local_latest["dt"])
        expected_earliest_dt = int(local_df["dt"].min())

        logger.info(
            f"Offline remote rows: {remote_rows} (expected {len(local_df)}), "
            f"dt span: {earliest_remote_dt} -> {latest_remote_dt}"
        )

        # 2. Query online store / latest feature vector
        vector, meta = self.connector.get_latest_feature_vector()
        online_latest_dt = meta.get("dt")
        online_verified = bool(online_latest_dt == expected_latest_dt)

        logger.info(
            f"Connector get_latest_feature_vector returned dt={online_latest_dt} "
            f"(matches expected latest {expected_latest_dt}: {online_verified})."
        )

        # 3. Project local latest row and verify numerical parity with online vector
        projected_local = self.connector.project_inference_features(
            pd.DataFrame([local_latest])
        )
        local_vector = projected_local.values.astype(np.float64)

        diff = np.abs(vector.reshape(-1) - local_vector.reshape(-1))
        max_diff = float(np.max(diff))
        parity_pass = bool(max_diff < 1e-4)

        # 4. Evaluate staleness
        current_epoch = time.time()
        age_hours = round((current_epoch - latest_remote_dt) / 3600.0, 2)
        is_stale = bool(age_hours > 3.0)

        logger.info(
            f"Remote verification: offline_rows={remote_rows}, columns={remote_columns}, "
            f"latest_dt={latest_remote_dt}, online_dt={online_latest_dt}, "
            f"114-parity max_diff={max_diff:.8f}, staleness age={age_hours}h (is_stale={is_stale})."
        )

        return {
            "remote_offline_row_count": remote_rows,
            "remote_column_count": remote_columns,
            "earliest_remote_dt": earliest_remote_dt,
            "latest_remote_dt": latest_remote_dt,
            "earliest_expected_dt": expected_earliest_dt,
            "latest_expected_dt": expected_latest_dt,
            "online_latest_dt": online_latest_dt,
            "online_verified": online_verified,
            "parity_max_difference": max_diff,
            "parity_pass": parity_pass,
            "input_age_hours": age_hours,
            "is_stale": is_stale,
        }

    def verify_idempotency(self, sample_size: int = 50) -> dict[str, Any]:
        """Verify that re-inserting existing historical rows does not inflate offline row count.

        Args:
            sample_size: Number of historical rows to re-insert.

        Returns:
            Idempotency test result summary.
        """
        logger.info(f"Executing idempotency check with {sample_size} historical rows...")
        df = pd.read_csv(self.csv_path)
        sample = df.iloc[:sample_size].copy()

        fs = self.connector._login()
        fg = fs.get_feature_group(
            name=self.connector.feature_group_name,
            version=self.connector.feature_group_version,
        )

        # Count before
        count_before = len(fg.select_all().filter(fg.location_id == self.connector.location_id).read())
        logger.info(f"Row count before re-insertion: {count_before}")

        # Wait for idle and re-insert sample
        wait_for_materialization_idle(fg, timeout_seconds=300, poll_interval=10)
        self.connector.insert_features(sample, wait=False, storage="offline")

        # Wait for materialization
        logger.info("Waiting for materialization of idempotency sample...")
        status = wait_for_materialization_idle(fg, timeout_seconds=600, poll_interval=15)
        logger.info(f"Idempotency materialization finished with status: {status}")

        # Count after
        count_after = len(fg.select_all().filter(fg.location_id == self.connector.location_id).read())
        logger.info(f"Row count after re-insertion: {count_after}")

        idempotent = bool(count_before == count_after)
        return {
            "sample_size": sample_size,
            "count_before": count_before,
            "count_after": count_after,
            "idempotent": idempotent,
        }


def main():
    """CLI entry point for Hopsworks historical feature backfill."""
    parser = argparse.ArgumentParser(
        description="Pearls AQI Predictor - Hopsworks Feature Store Backfill"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local dataset, schema, and chunks without connecting to cloud.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of rows per chunk during ingestion (default: 50000).",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait synchronously for ingestion job completion.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="offline",
        choices=["offline", "online", "all"],
        help="Target storage for ingestion ('offline' [default], 'online', or 'all').",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-backfill verification.",
    )
    parser.add_argument(
        "--no-sync-online",
        action="store_true",
        help="Skip synchronizing latest observation to online store.",
    )
    parser.add_argument(
        "--idempotency-check",
        action="store_true",
        help="Run idempotency verification by re-inserting historical rows.",
    )

    args = parser.parse_args()

    runner = HopsworksBackfillRunner(
        chunk_size=args.chunk_size,
        wait=not args.no_wait,
        storage=args.storage,
    )

    try:
        if args.idempotency_check:
            res = runner.verify_idempotency()
        else:
            res = runner.run(
                dry_run=args.dry_run,
                verify_after=not args.no_verify,
                sync_online=not args.no_sync_online,
            )
        print("\n" + "=" * 60)
        print("HOPSWORKS BACKFILL EXECUTION RESULT")
        print("=" * 60)
        print(json.dumps(res, indent=2))
        print("=" * 60)
    except Exception as e:
        logger.error(f"Backfill execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
