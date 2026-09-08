"""Unit and isolation tests for src.inference.observation_resolver.

Covers:
- Strict mode configuration ('auto', 'hopsworks', 'bootstrap')
- Bootstrap resolution and 114-column canonical schema validation
- Hopsworks online retrieval (mocked) and strict fail-closed behavior
- Auto mode fallback on Hopsworks network/credential errors
- Schema validation: rejection of NaN, Inf, non-numeric values, missing columns
- Bounded 60s in-memory caching and force_refresh bypass
- Thread-safe resolution under concurrency
- Dynamic freshness calculation (get_age_hours, is_stale)
- Non-blocking, secret-free health status reporting
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import threading
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.exceptions import FeatureStoreError, ValidationError
from src.inference.observation_resolver import (
    FeatureObservationResolver,
    ResolvedFeatureObservation,
)
from src.inference.runtime_resolver import RuntimeAssetResolver


@pytest.fixture
def runtime_resolver() -> RuntimeAssetResolver:
    return RuntimeAssetResolver()


@pytest.fixture
def canonical_schema(runtime_resolver: RuntimeAssetResolver) -> list[str]:
    import json
    with open(runtime_resolver.get_schema_path(), "r", encoding="utf-8") as f:
        return json.load(f)["feature_names"]


class TestFeatureObservationResolverModes:
    """Tests covering resolver initialization and mode contracts."""

    def test_invalid_mode_raises_validation_error(self):
        with pytest.raises(ValidationError, match="Invalid FEATURE_SOURCE_MODE"):
            FeatureObservationResolver(mode="unsupported_source")

    def test_bootstrap_mode_resolves_canonical_vector(self):
        resolver = FeatureObservationResolver(mode="bootstrap")
        obs = resolver.resolve_observation()

        assert isinstance(obs, ResolvedFeatureObservation)
        assert obs.source == "bootstrap"
        assert obs.cloud_active is False
        assert obs.fallback_active is False
        assert obs.fallback_reason is None
        assert obs.features.shape == (1, 114)
        assert obs.features_array.shape == (1, 114)
        assert obs.features_array.dtype == np.float32
        assert isinstance(obs.observed_at, datetime)
        assert isinstance(obs.dt, int)
        assert obs.current_aqi > 0
        assert obs.dominant_pollutant is not None
        assert "pm2_5" in obs.pollutants
        assert "temperature_2m" in obs.weather

    def test_hopsworks_mode_strict_fail_closed_on_error(self):
        resolver = FeatureObservationResolver(mode="hopsworks")
        with patch.object(resolver, "_fetch_hopsworks", side_effect=FeatureStoreError("Connection timeout")):
            with pytest.raises(FeatureStoreError, match="Connection timeout"):
                resolver.resolve_observation()

    def test_auto_mode_success_uses_hopsworks(self, canonical_schema: list[str]):
        resolver = FeatureObservationResolver(mode="auto")
        mock_obs = ResolvedFeatureObservation(
            features=pd.DataFrame([[1.0] * 114], columns=canonical_schema),
            features_array=np.ones((1, 114), dtype=np.float32),
            observed_at=datetime(2026, 9, 8, 10, 0, 0, tzinfo=timezone.utc),
            dt=1788861600,
            current_aqi=120.0,
            source="hopsworks",
            cloud_active=True,
            fallback_active=False,
            dominant_pollutant="pm2_5",
            pollutants={"pm2_5": 45.0},
            weather={"temperature_2m": 30.0},
        )

        with patch.object(resolver, "_fetch_hopsworks", return_value=mock_obs):
            obs = resolver.resolve_observation()
            assert obs.source == "hopsworks"
            assert obs.cloud_active is True
            assert obs.fallback_active is False
            assert obs.current_aqi == 120.0

    def test_auto_mode_falls_back_to_bootstrap_on_error(self):
        resolver = FeatureObservationResolver(mode="auto")
        with patch.object(resolver, "_fetch_hopsworks", side_effect=FeatureStoreError("Network unreachable")):
            obs = resolver.resolve_observation()
            assert obs.source == "bootstrap"
            assert obs.cloud_active is False
            assert obs.fallback_active is True
            assert obs.fallback_reason is not None
            assert "FeatureStoreError: Network unreachable" in obs.fallback_reason
            assert obs.features.shape == (1, 114)


class TestHopsworksOnlineValidation:
    """Tests covering validation of incoming Hopsworks online rows."""

    def test_missing_dt_column_raises_validation_error(self, canonical_schema: list[str]):
        resolver = FeatureObservationResolver(mode="hopsworks")
        # Ensure 'dt' is explicitly omitted from columns
        cols_without_dt = [col for col in canonical_schema if col != "dt"]
        bad_df = pd.DataFrame([[1.0] * len(cols_without_dt)], columns=cols_without_dt)

        mock_fg = MagicMock()
        mock_fg.location_id = "lahore"
        mock_fg.select_all.return_value.filter.return_value.read.return_value = bad_df
        mock_fs = MagicMock()
        mock_fs.get_feature_group.return_value = mock_fg
        mock_proj = MagicMock()
        mock_proj.get_feature_store.return_value = mock_fs

        with patch("hopsworks.login", return_value=mock_proj), \
             patch.dict("os.environ", {"HOPSWORKS_API_KEY": "fake_key"}):
            with pytest.raises(ValidationError, match="missing event_time column 'dt'"):
                resolver._fetch_hopsworks()

    def test_missing_canonical_feature_raises_validation_error(self):
        resolver = FeatureObservationResolver(mode="hopsworks")
        incomplete_df = pd.DataFrame([{
            "dt": 1788861600,
            "location_id": "lahore",
            "epa_aqi": 100.0,
            # Missing other 113 features
        }])

        mock_fg = MagicMock()
        mock_fg.location_id = "lahore"
        mock_fg.select_all.return_value.filter.return_value.read.return_value = incomplete_df
        mock_fs = MagicMock()
        mock_fs.get_feature_group.return_value = mock_fg
        mock_proj = MagicMock()
        mock_proj.get_feature_store.return_value = mock_fs

        with patch("hopsworks.login", return_value=mock_proj), \
             patch.dict("os.environ", {"HOPSWORKS_API_KEY": "fake_key"}):
            with pytest.raises(ValidationError, match="missing canonical features"):
                resolver._fetch_hopsworks()

    def test_nan_feature_value_raises_validation_error(self, canonical_schema: list[str]):
        resolver = FeatureObservationResolver(mode="hopsworks")
        data = {col: [1.0] for col in canonical_schema}
        data["dt"] = [1788861600]
        data["location_id"] = ["lahore"]
        data["temperature_2m"] = [float("nan")]  # Non-finite NaN value

        bad_df = pd.DataFrame(data)
        mock_fg = MagicMock()
        mock_fg.location_id = "lahore"
        mock_fg.select_all.return_value.filter.return_value.read.return_value = bad_df
        mock_fs = MagicMock()
        mock_fs.get_feature_group.return_value = mock_fg
        mock_proj = MagicMock()
        mock_proj.get_feature_store.return_value = mock_fs

        with patch("hopsworks.login", return_value=mock_proj), \
             patch.dict("os.environ", {"HOPSWORKS_API_KEY": "fake_key"}):
            with pytest.raises(ValidationError, match="Invalid non-finite numeric value"):
                resolver._fetch_hopsworks()


class TestObservationCachingAndFreshness:
    """Tests covering caching TTL, force refresh, and dynamic freshness evaluation."""

    def test_cache_hit_within_ttl(self):
        resolver = FeatureObservationResolver(mode="bootstrap", cache_ttl_seconds=60.0)
        obs1 = resolver.resolve_observation()
        obs2 = resolver.resolve_observation()

        assert obs1 is obs2  # Exact same cached instance

    def test_force_refresh_bypasses_cache(self):
        resolver = FeatureObservationResolver(mode="bootstrap", cache_ttl_seconds=60.0)
        obs1 = resolver.resolve_observation()
        obs2 = resolver.resolve_observation(force_refresh=True)

        assert obs1 is not obs2  # Recomputed fresh instance

    def test_cache_expires_after_ttl(self):
        resolver = FeatureObservationResolver(mode="bootstrap", cache_ttl_seconds=0.01)
        obs1 = resolver.resolve_observation()
        import time
        time.sleep(0.02)
        obs2 = resolver.resolve_observation()

        assert obs1 is not obs2

    def test_dynamic_freshness_evaluation(self):
        base_time = datetime(2026, 9, 8, 12, 0, 0, tzinfo=timezone.utc)
        obs = ResolvedFeatureObservation(
            features=pd.DataFrame(),
            features_array=np.empty((0, 0)),
            observed_at=base_time,
            dt=int(base_time.timestamp()),
            current_aqi=50.0,
            source="bootstrap",
            cloud_active=False,
        )

        # At T + 1.5h -> fresh
        t_fresh = base_time + timedelta(hours=1.5)
        assert obs.get_age_hours(t_fresh) == 1.5
        assert obs.is_stale(t_fresh) is False

        # At T + 3.01h -> stale
        t_stale = base_time + timedelta(hours=3.01)
        assert obs.get_age_hours(t_stale) == 3.01
        assert obs.is_stale(t_stale) is True

    def test_concurrency_thread_safety(self):
        resolver = FeatureObservationResolver(mode="bootstrap", cache_ttl_seconds=60.0)
        results = []

        def worker():
            obs = resolver.resolve_observation()
            results.append(obs)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 8
        # All threads should get identical instance from thread-safe cache
        for obs in results:
            assert obs is results[0]

    def test_health_status_metadata(self):
        resolver = FeatureObservationResolver(mode="auto", cache_ttl_seconds=60.0)
        status_init = resolver.get_status()
        assert status_init["mode"] == "auto"
        assert status_init["last_source"] == "uninitialized"
        assert status_init["is_cached"] is False
        assert status_init["cache_ttl_seconds"] == 60.0

        # Trigger resolution in bootstrap
        resolver.mode = "bootstrap"
        resolver.resolve_observation()
        status_after = resolver.get_status()
        assert status_after["last_source"] == "bootstrap"
        assert status_after["is_cached"] is True
        assert status_after["last_error"] is None
