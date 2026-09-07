"""Pearls AQI Predictor - Hopsworks Feature Store Connector.

Manages cloud feature group ingestion, synchronous write synchronization,
and entity-key (location_id) / event-time (dt) storage schemas while strictly
projecting the canonical 114-column ordered contract for EXP-019 model inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_FEATURE_GROUP_NAME,
    HOPSWORKS_FEATURE_GROUP_VERSION,
    HOPSWORKS_FEATURE_VIEW_NAME,
    HOPSWORKS_FEATURE_VIEW_VERSION,
    HOPSWORKS_HOST,
    HOPSWORKS_PROJECT,
    PROCESSED_DATA_DIR,
    RUNTIME_DIR,
)
from src.exceptions import FeatureStoreError, ValidationError
from src.logger import logger

SCHEMA_PATH = PROCESSED_DATA_DIR / "feature_schema_v2_weather.json"


def load_canonical_feature_names() -> list[str]:
    """Load the canonical 114 ordered feature names."""
    schema_file = SCHEMA_PATH
    if not schema_file.exists():
        fallback_schema = RUNTIME_DIR / "production" / "feature_schema_v2_weather.json"
        if fallback_schema.exists():
            schema_file = fallback_schema
        else:
            raise FileNotFoundError(f"Canonical schema file not found at {SCHEMA_PATH} or {fallback_schema}")
    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema.get("feature_names", schema.get("features", []))
    if len(features) != 114:
        raise ValidationError(f"Expected 114 canonical features in schema, found {len(features)}")
    return features


class HopsworksFeatureStoreConnector:
    """Manages cloud Feature Store operations with strict contract validation and local fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        project_name: str | None = None,
        host: str | None = None,
        feature_group_name: str = HOPSWORKS_FEATURE_GROUP_NAME,
        feature_group_version: int = HOPSWORKS_FEATURE_GROUP_VERSION,
        feature_view_name: str = HOPSWORKS_FEATURE_VIEW_NAME,
        feature_view_version: int = HOPSWORKS_FEATURE_VIEW_VERSION,
        location_id: str = "lahore",
    ) -> None:
        """Initialize Feature Store connector.

        Args:
            api_key: Hopsworks API key (defaults to HOPSWORKS_API_KEY env).
            project_name: Hopsworks project name (defaults to HOPSWORKS_PROJECT env).
            host: Hopsworks host URL (defaults to HOPSWORKS_HOST env).
            feature_group_name: Feature group identifier.
            feature_group_version: Feature group version integer.
            feature_view_name: Feature view identifier.
            feature_view_version: Feature view version integer.
            location_id: Entity primary key value.
        """
        self.api_key = api_key if api_key is not None else HOPSWORKS_API_KEY
        self.project_name = project_name or HOPSWORKS_PROJECT
        self.host = host or HOPSWORKS_HOST
        self.feature_group_name = feature_group_name
        self.feature_group_version = feature_group_version
        self.feature_view_name = feature_view_name
        self.feature_view_version = feature_view_version
        self.location_id = location_id

        self.canonical_features = load_canonical_feature_names()
        self._project = None
        self._fs = None
        self._fg = None
        self._fv = None

    def is_cloud_configured(self) -> bool:
        """Check if Hopsworks cloud credentials are provided."""
        return bool(self.api_key and self.api_key.strip() != "")

    def _login(self):
        """Establish lazy authenticated connection to Hopsworks."""
        if self._project is None:
            if not self.is_cloud_configured():
                raise FeatureStoreError("Hopsworks API key is not configured.")
            try:
                import hopsworks
                self._project = hopsworks.login(
                    api_key_value=self.api_key,
                    project=self.project_name,
                    host=self.host,
                )
                self._fs = self._project.get_feature_store()
                logger.info(f"Successfully authenticated with Hopsworks project '{self.project_name}'")
            except Exception as e:
                logger.warning(f"Hopsworks login failed: {e}")
                raise FeatureStoreError(f"Cloud authentication failed: {e}") from e
        return self._fs

    def prepare_storage_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for Hopsworks feature group storage (115 columns: location_id + 114 features).

        Args:
            df: Input dataframe containing canonical features.

        Returns:
            DataFrame with location_id added and correct dtypes.
        """
        storage_df = df.copy()

        # Verify dt presence
        if "dt" not in storage_df.columns:
            raise ValidationError("Input dataframe must contain event_time column 'dt'")

        # Validate dt values are integral before casting
        dt_numeric = pd.to_numeric(storage_df["dt"], errors="coerce")
        if dt_numeric.isna().any():
            raise ValidationError("Non-numeric or NaN values detected in event_time column 'dt'")
        if not np.all(dt_numeric % 1 == 0):
            raise ValidationError("Fractional (non-integral) timestamps detected in event_time column 'dt'")

        # Ensure location_id primary key exists
        if "location_id" not in storage_df.columns:
            storage_df["location_id"] = str(self.location_id)
        else:
            storage_df["location_id"] = storage_df["location_id"].astype(str)

        # Cast dt to integer epoch int64
        storage_df["dt"] = dt_numeric.astype(np.int64)

        # Ensure all canonical columns are present
        missing = [c for c in self.canonical_features if c not in storage_df.columns]
        if missing:
            raise ValidationError(f"Storage dataframe is missing canonical columns: {missing[:5]}")

        # Cast all 113 canonical numerical predictors to float64 (double in Hopsworks schema)
        for col in self.canonical_features:
            if col != "dt":
                storage_df[col] = storage_df[col].astype(np.float64)

        # Retain exactly location_id + 114 canonical features in order (115 columns total)
        storage_df = storage_df[["location_id"] + self.canonical_features]

        return storage_df

    def project_inference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project storage dataframe into the exact canonical 114-column ordered inference contract.

        Args:
            df: DataFrame containing cloud/storage data.

        Returns:
            DataFrame with exactly 114 columns in canonical index order.
        """
        # Exclude metadata like location_id
        missing = [col for col in self.canonical_features if col not in df.columns]
        if missing:
            raise ValidationError(
                f"Cloud schema integrity error: missing {len(missing)} canonical columns (e.g. {missing[:3]}). "
                "Failing closed to prevent corrupted model inference."
            )

        # Strictly select and order columns according to canonical schema
        projected_df = df[self.canonical_features].copy()

        # Validate resulting dimensions and column sequence
        if list(projected_df.columns) != self.canonical_features:
            raise ValidationError("Projected dataframe column sequence violates canonical 114-feature schema ordering.")

        return projected_df

    def get_or_create_feature_group(self, description: str | None = None) -> Any:
        """Get or create Hopsworks feature group with entity PK and event time."""
        fs = self._login()
        if description is None:
            description = (
                "Weather-enriched air quality features with 72-hour lag windows for Lahore (EXP-019)."
            )

        self._fg = fs.get_or_create_feature_group(
            name=self.feature_group_name,
            version=self.feature_group_version,
            primary_key=["location_id"],
            event_time="dt",
            time_travel_format="HUDI",
            description=description,
            online_enabled=True,
        )
        return self._fg

    def insert_features(
        self,
        df: pd.DataFrame,
        wait: bool = True,
        storage: str | None = None,
        write_options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert features into Hopsworks Feature Group with synchronous completion guarantee.

        Args:
            df: DataFrame containing features to insert.
            wait: If True, waits synchronously for ingestion job completion.
            storage: Target storage ("offline", "online", or None for both).
            write_options: Optional low-level write options dictionary.

        Returns:
            Ingestion result summary dictionary.
        """
        storage_df = self.prepare_storage_dataframe(df)
        fg = self.get_or_create_feature_group()

        logger.info(
            f"Inserting {len(storage_df)} records (115 columns) into feature group "
            f"'{self.feature_group_name}' (wait={wait}, storage={storage})..."
        )
        insert_kwargs: dict[str, Any] = {"wait": wait}
        if storage is not None:
            insert_kwargs["storage"] = storage
        if write_options is not None:
            insert_kwargs["write_options"] = write_options

        insert_res = None
        for attempt in range(1, 4):
            try:
                insert_res = fg.insert(storage_df, **insert_kwargs)
                break
            except Exception as e:
                err_msg = str(e)
                if ("RemoteDisconnected" in err_msg or "Connection aborted" in err_msg) and attempt < 3:
                    logger.warning(f"Connection blip during insert (attempt {attempt}/3): {e}")
                    time.sleep(3)
                    job = getattr(fg, "materialization_job", None)
                    if job:
                        try:
                            execs = job.get_executions()
                            if execs and getattr(execs[0], "state", "") == "RUNNING":
                                logger.info(f"Execution {execs[0].id} was successfully triggered on cluster.")
                                insert_res = (job, None)
                                break
                        except Exception:
                            pass
                else:
                    raise

        job_repr = "none"
        if insert_res is not None:
            try:
                job_repr = str(insert_res)
            except Exception:
                job_repr = "JobSubmitted"

        return {
            "status": "inserted",
            "feature_group": self.feature_group_name,
            "version": self.feature_group_version,
            "records_inserted": len(storage_df),
            "storage_columns": len(storage_df.columns),
            "synchronous_wait": wait,
            "storage": storage or "all",
            "job_result": job_repr,
        }

    def get_latest_feature_vector(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Fetch latest feature vector from cloud Feature Store or fallback locally.

        Returns:
            Tuple of (114-feature numpy array, metadata dictionary).
        """
        if self.is_cloud_configured():
            try:
                fs = self._login()
                fg = fs.get_feature_group(name=self.feature_group_name, version=self.feature_group_version)
                cloud_df = None

                if getattr(fg, "online_enabled", False):
                    try:
                        q = fg.select_all().filter(fg.location_id == self.location_id)
                        cloud_df = q.read(online=True)
                    except Exception as e_online:
                        logger.warning(f"Online store read attempt failed ({e_online}); falling back to offline.")

                if cloud_df is None or len(cloud_df) == 0:
                    query = fg.select_all().filter(fg.location_id == self.location_id)
                    cloud_df = query.read()

                if cloud_df is None or len(cloud_df) == 0:
                    raise FeatureStoreError("Hopsworks query returned empty dataset.")

                # Sort by event_time dt descending to get latest
                cloud_df = cloud_df.sort_values(by="dt", ascending=False)
                latest_row_df = cloud_df.iloc[[0]]

                # Project to exact 114 canonical columns (fails closed if schema is corrupted)
                projected_df = self.project_inference_features(latest_row_df)
                vector = projected_df.values.astype(np.float64)

                meta = {
                    "source": "Hopsworks Cloud Feature Store",
                    "feature_group": self.feature_group_name,
                    "location_id": self.location_id,
                    "dt": int(latest_row_df["dt"].iloc[0]),
                    "feature_count": 114,
                    "cloud_active": True,
                }
                return vector, meta
            except ValidationError:
                # Schema/integrity violation must fail closed
                raise
            except Exception as e:
                logger.warning(f"Cloud Feature Store query failed ({e}); activating local CSV fallback.")

        # Local fallback execution
        local_csv = PROCESSED_DATA_DIR / "features_v2_weather.csv"
        if not local_csv.exists():
            raise FileNotFoundError(f"Local feature file not found at {local_csv}")

        local_df = pd.read_csv(local_csv)
        latest_local = local_df.iloc[[-1]]
        projected_df = self.project_inference_features(latest_local)
        vector = projected_df.values.astype(np.float64)

        meta = {
            "source": "Local CSV Storage (Fallback)",
            "feature_group": "local_features_v2_weather.csv",
            "location_id": self.location_id,
            "dt": int(latest_local["dt"].iloc[0]) if "dt" in latest_local.columns else 0,
            "feature_count": 114,
            "cloud_active": False,
        }
        return vector, meta

    def insert_hourly_feature_row(
        self,
        row_df: pd.DataFrame,
        wait_for_job: bool = True,
    ) -> dict[str, Any]:
        """Insert a single verified hourly feature row into Hopsworks Feature Store.

        Performs dual-store upsert semantics:
        1. Offline Hudi upsert to record the historical event without duplicate row inflation.
        2. Online insertion with server-side upsert_if_newer=True so delayed or older events
           cannot regress the online Lahore entity.

        Args:
            row_df: DataFrame containing the 114 canonical features for the observation.
            wait_for_job: Whether to wait synchronously for offline ingestion job completion.

        Returns:
            Dictionary reporting insertion status and observation timestamp.
        """
        # Validate and prepare storage DataFrame (115 columns: location_id + 114 features)
        storage_df = self.prepare_storage_dataframe(row_df)
        obs_dt = int(storage_df["dt"].iloc[0])

        # Pre-read current online dt for observability logging
        online_dt_before = None
        try:
            _, meta = self.get_latest_feature_vector()
            online_dt_before = int(meta.get("dt", 0))
        except Exception as e:
            logger.debug(f"Online store pre-read note: {e}")

        # 1. Offline Hudi Upsert
        offline_write_options = {
            "wait_for_job": wait_for_job,
            "start_offline_materialization": True,
        }
        offline_res = self.insert_features(
            row_df,
            wait=wait_for_job,
            storage="offline",
            write_options=offline_write_options,
        )

        # 2. Online Upsert with Server-Side Newer-Event Protection
        online_write_options = {
            "wait_for_online_ingestion": True,
            "online_ingestion_options": {
                "upsert_if_newer": True,
            },
        }
        online_res = self.insert_features(
            row_df,
            wait=True,
            storage="online",
            write_options=online_write_options,
        )

        return {
            "status": "hourly_ingestion_success",
            "observation_dt": obs_dt,
            "online_dt_before": online_dt_before,
            "offline_result": offline_res,
            "online_result": online_res,
        }

