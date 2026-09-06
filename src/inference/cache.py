"""Pearls AQI Predictor - Prediction Cache.

File-based local caching layer with Time-To-Live (TTL) support to eliminate
redundant inference computation across repeated API or dashboard queries.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.config import DATA_DIR
from src.logger import logger


class PredictionCache:
    """Provides file-backed caching of predictions with expiration logic."""

    def __init__(self, cache_file: Path | str | None = None) -> None:
        """Initialize cache store.

        Args:
            cache_file: Path to destination JSON cache file (default: data/cached_predictions.json).
        """
        self.cache_file = Path(cache_file or DATA_DIR / "cached_predictions.json")
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def _read_cache(self) -> dict[str, Any]:
        """Read all entries from cache file."""
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read prediction cache from {self.cache_file}: {e}")
            return {}

    def _write_cache(self, data: dict[str, Any]) -> None:
        """Write all entries to cache file."""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write prediction cache to {self.cache_file}: {e}")

    def get(self, key: str, ttl_seconds: int = 3600) -> dict[str, Any] | None:
        """Retrieve a cached prediction entry if it exists and has not expired.

        Args:
            key: Cache key string (e.g. 'latest_72h_forecast').
            ttl_seconds: Maximum allowed age in seconds (default: 3600s = 1h).

        Returns:
            Cached data dictionary if valid, None otherwise.
        """
        cache = self._read_cache()
        entry = cache.get(key)
        if not entry:
            return None

        cached_at_str = entry.get("cached_at")
        if not cached_at_str:
            return None

        try:
            cached_at = datetime.fromisoformat(cached_at_str)
            now = datetime.now(timezone.utc)
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)

            age = (now - cached_at).total_seconds()
            if age > ttl_seconds:
                logger.debug(f"Cache key '{key}' expired (age {age:.1f}s > TTL {ttl_seconds}s).")
                return None

            logger.info(f"Cache hit for key '{key}' (age {age:.1f}s).")
            return entry.get("payload")
        except Exception as e:
            logger.warning(f"Error checking cache expiration for '{key}': {e}")
            return None

    def set(self, key: str, payload: dict[str, Any]) -> None:
        """Store a prediction entry in the cache with the current UTC timestamp.

        Args:
            key: Cache key string.
            payload: Prediction data dictionary to persist.
        """
        cache = self._read_cache()
        now = datetime.now(timezone.utc)
        cache[key] = {
            "cached_at": now.isoformat(),
            "payload": payload,
        }
        self._write_cache(cache)
        logger.info(f"Persisted prediction cache entry for key '{key}'.")

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._write_cache({})
        logger.info("Cleared prediction cache.")
