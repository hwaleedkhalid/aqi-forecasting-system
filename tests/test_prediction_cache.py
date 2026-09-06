"""Unit tests for src.inference.cache."""

from __future__ import annotations

from pathlib import Path
import time
import pytest

from src.inference.cache import PredictionCache


class TestPredictionCache:
    """Test suite for PredictionCache."""

    def test_cache_miss_on_empty(self, tmp_path: Path):
        cache = PredictionCache(cache_file=tmp_path / "cache.json")
        assert cache.get("nonexistent_key") is None

    def test_cache_set_and_get_hit(self, tmp_path: Path):
        cache = PredictionCache(cache_file=tmp_path / "cache.json")
        payload = {"data": [1, 2, 3], "status": "ok"}
        cache.set("test_key", payload)

        retrieved = cache.get("test_key", ttl_seconds=60)
        assert retrieved == payload

    def test_cache_expiration_on_ttl(self, tmp_path: Path):
        cache = PredictionCache(cache_file=tmp_path / "cache.json")
        cache.set("short_key", {"msg": "hello"})

        # Immediate retrieve with 0s TTL must expire
        expired = cache.get("short_key", ttl_seconds=0)
        assert expired is None

    def test_cache_clear(self, tmp_path: Path):
        cache = PredictionCache(cache_file=tmp_path / "cache.json")
        cache.set("key1", {"val": 1})
        cache.set("key2", {"val": 2})
        assert cache.get("key1") is not None

        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_corrupted_cache_file_handles_gracefully(self, tmp_path: Path):
        cache_file = tmp_path / "corrupted.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write("NOT_VALID_JSON{{{")

        cache = PredictionCache(cache_file=cache_file)
        assert cache.get("any_key") is None
