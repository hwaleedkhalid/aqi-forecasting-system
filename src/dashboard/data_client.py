"""Pearls AQI Predictor - Dashboard Data Client.

Unified client for Streamlit that queries the Flask REST API with seamless,
fully-contract-parity fallback to direct local inference when the API service
is offline or unreachable.
"""

from __future__ import annotations

import os
from typing import Any
import requests

from src.inference.predictor import AQIPredictor
from src.logger import logger

DEFAULT_API_URL = os.environ.get("FLASK_API_URL", "http://127.0.0.1:5000/api")


class DashboardDataClient:
    """Client fetching dashboard data from Flask REST API or direct inference fallback."""

    def __init__(
        self,
        api_base_url: str = DEFAULT_API_URL,
        timeout: float = 2.0,
        predictor: AQIPredictor | None = None,
    ) -> None:
        """Initialize data client.

        Args:
            api_base_url: Base URL for Flask REST API (default: http://127.0.0.1:5000/api).
            timeout: HTTP request timeout in seconds.
            predictor: Optional AQIPredictor instance for fallback execution.
        """
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self._predictor = predictor

    @property
    def predictor(self) -> AQIPredictor:
        """Lazy-loaded AQIPredictor for direct inference fallback."""
        if self._predictor is None:
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

    def fetch_current(self) -> tuple[dict[str, Any], str]:
        """Fetch latest observed telemetry.

        Returns:
            Tuple of (data_dict, source_mode_string).
        """
        try:
            resp = requests.get(f"{self.api_base_url}/current", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json(), "REST API"
            logger.warning(f"API /current returned HTTP {resp.status_code}, activating fallback.")
        except Exception as e:
            logger.debug(f"API connection failed for /current ({e}), activating direct fallback.")

        # Fallback to direct local observation reader
        data = self.predictor.get_latest_observation()
        return data, "Direct Local Inference (API Offline Fallback)"

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
            logger.warning(f"API /forecast returned HTTP {resp.status_code}, activating fallback.")
        except Exception as e:
            logger.debug(f"API connection failed for /forecast ({e}), activating direct fallback.")

        # Fallback to direct local model inference
        data = self.predictor.predict_latest(use_cache=True, force_refresh=force_refresh)
        return data, "Direct Local Inference (API Offline Fallback)"

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
            logger.debug(f"API connection failed for /model/info ({e}), activating direct fallback.")

        # Fallback to direct local metadata
        data = self.predictor.get_model_metadata()
        return data, "Direct Local Inference (API Offline Fallback)"

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
            logger.warning(f"API /explain returned HTTP {resp.status_code}, activating fallback.")
        except Exception as e:
            logger.debug(f"API connection failed for /explain ({e}), activating direct fallback.")

        # Fallback to direct local explainer
        data = self.predictor.explain_latest(horizon=horizon, top_k=top_k)
        return data, "Direct Local Inference (API Offline Fallback)"

