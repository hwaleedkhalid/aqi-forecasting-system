"""Pearls AQI Predictor - Dashboard Data Client.

Unified client for Streamlit that queries the Flask REST API with seamless,
fully-contract-parity fallback to direct local inference when the API service
is offline or unreachable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import requests

# Ensure project root and dashboard directory are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

try:
    from src.logger import logger
except Exception:
    import logging

    logger = logging.getLogger("pearls_aqi")

DEFAULT_API_URL = os.environ.get("FLASK_API_URL", "http://127.0.0.1:5000/api")
ENABLE_LOCAL_FALLBACK = os.environ.get("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"
DEFAULT_TIMEOUT = float(os.environ.get("API_TIMEOUT_SECONDS", "10.0"))


class DashboardDataClient:
    """Client fetching dashboard data from Flask REST API or direct inference fallback."""

    def __init__(
        self,
        api_base_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        predictor: Any | None = None,
        enable_local_fallback: bool | None = None,
    ) -> None:
        """Initialize data client.

        Args:
            api_base_url: Base URL for Flask REST API (default: http://127.0.0.1:5000/api).
            timeout: HTTP request timeout in seconds.
            predictor: Optional AQIPredictor instance for fallback execution.
            enable_local_fallback: Whether to permit falling back to in-process AQIPredictor.
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self._predictor = predictor
        if enable_local_fallback is not None:
            self.enable_local_fallback = enable_local_fallback
        elif predictor is not None:
            self.enable_local_fallback = True
        else:
            self.enable_local_fallback = ENABLE_LOCAL_FALLBACK

    @property
    def predictor(self) -> Any:
        """Lazy-loaded AQIPredictor for direct inference fallback."""
        if self._predictor is None:
            if not self.enable_local_fallback:
                raise RuntimeError(
                    "Local inference fallback is disabled (ENABLE_LOCAL_FALLBACK=false). "
                    "Cannot initialize in-process AQIPredictor."
                )
            from src.inference.predictor import AQIPredictor
            self._predictor = AQIPredictor()
        return self._predictor

    def is_api_online(self) -> bool:
        """Check if Flask REST API is online and reports healthy."""
        try:
            resp = requests.get(f"{self.api_base_url}/health", timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return bool(data.get("service_ready", False))
            return False
        except Exception:
            return False

    def _handle_fallback(self, method_name: str, *args: Any, **kwargs: Any) -> tuple[dict[str, Any], str]:
        """Execute local fallback if enabled, or raise user-friendly connection error."""
        if not self.enable_local_fallback:
            raise ConnectionError(
                f"The forecasting service is starting from standby or unreachable at {self.api_base_url}. "
                "Please retry in a few moments."
            )
        method = getattr(self.predictor, method_name)
        data = method(*args, **kwargs)
        return data, "Direct Local Inference (API Offline Fallback)"

    def fetch_current(self) -> tuple[dict[str, Any], str]:
        """Fetch latest observed telemetry.

        Returns:
            Tuple of (data_dict, source_mode_string).
        """
        try:
            resp = requests.get(f"{self.api_base_url}/current", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json(), "REST API"
            logger.warning(f"API /current returned HTTP {resp.status_code}, attempting fallback.")
        except Exception as e:
            logger.debug(f"API connection failed for /current ({e}), attempting fallback.")

        return self._handle_fallback("get_latest_observation")

    def fetch_forecast(self, force_refresh: bool = False) -> tuple[dict[str, Any], str]:
        """Fetch 72-hour AQI forecast.

        Args:
            force_refresh: If True, bypasses prediction cache and recomputes.

        Returns:
            Tuple of (forecast_dict, source_mode_string).
        """
        param_str = "true" if force_refresh else "false"
        try:
            resp = requests.get(
                f"{self.api_base_url}/forecast",
                params={"force_refresh": param_str},
                timeout=self.timeout * 2,  # Allow slightly more time for model inference
            )
            if resp.status_code == 200:
                return resp.json(), "REST API"
            logger.warning(f"API /forecast returned HTTP {resp.status_code}, attempting fallback.")
        except Exception as e:
            logger.debug(f"API connection failed for /forecast ({e}), attempting fallback.")

        return self._handle_fallback("predict_latest", use_cache=True, force_refresh=force_refresh)

    def fetch_model_info(self) -> tuple[dict[str, Any], str]:
        """Fetch production model provenance and benchmarks.

        Returns:
            Tuple of (metadata_dict, source_mode_string).
        """
        try:
            resp = requests.get(f"{self.api_base_url}/model/info", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json(), "REST API"
        except Exception as e:
            logger.debug(f"API connection failed for /model/info ({e}), attempting fallback.")

        return self._handle_fallback("get_model_metadata")

    def fetch_explain(self, horizon: int = 24, top_k: int = 10) -> tuple[dict[str, Any], str]:
        """Fetch SHAP explanation and feature attribution for a specific horizon.

        Args:
            horizon: Target prediction horizon (1..72).
            top_k: Number of top features to return.

        Returns:
            Tuple of (explanation_dict, source_mode_string).
        """
        try:
            resp = requests.get(
                f"{self.api_base_url}/explain",
                params={"horizon": str(horizon), "top_k": str(top_k)},
                timeout=self.timeout * 3,
            )
            if resp.status_code == 200:
                return resp.json(), "REST API"
            logger.warning(f"API /explain returned HTTP {resp.status_code}, attempting fallback.")
        except Exception as e:
            logger.debug(f"API connection failed for /explain ({e}), attempting fallback.")

        return self._handle_fallback("explain_latest", horizon=horizon, top_k=top_k)

