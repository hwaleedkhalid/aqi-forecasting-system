"""Pearls AQI Predictor - Flask REST API Blueprint.

Defines HTTP routes for /api/health, /api/current, /api/forecast, and /api/model/info.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import Blueprint, current_app, jsonify, request
from werkzeug.exceptions import BadRequest, ServiceUnavailable

from src.exceptions import ValidationError
from src.inference.model_loader import ModelLoader
from src.inference.predictor import AQIPredictor
from src.logger import logger

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Lazy-loaded predictor singleton
_predictor: AQIPredictor | None = None


def get_predictor() -> AQIPredictor:
    """Retrieve or initialize the AQIPredictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = AQIPredictor()
    return _predictor


def parse_bool_arg(val: str | None, default: bool = False) -> bool:
    """Parse a boolean query argument strictly.

    Args:
        val: String parameter value from query string.
        default: Default boolean if parameter is omitted.

    Returns:
        Boolean value.

    Raises:
        BadRequest: If parameter is provided but not in {'true', 'false', '1', '0'}.
    """
    if val is None:
        return default

    clean_val = val.strip().lower()
    if clean_val in {"true", "1"}:
        return True
    if clean_val in {"false", "0"}:
        return False

    raise BadRequest(f"Invalid boolean query parameter '{val}': must be 'true', 'false', '1', or '0'.")


@api_bp.route("/health", methods=["GET"])
def health_check() -> Any:
    """Service liveness and model readiness check.

    Returns:
        200 OK with healthy status if model artifacts load.
        503 Service Unavailable with degraded status if model is missing or unready.
    """
    now = datetime.now(timezone.utc)
    try:
        predictor = get_predictor()
        # Verify model artifact readiness
        model = predictor.model_loader.load_model()
        scaler = predictor.model_loader.load_scaler()
        schema = predictor.model_loader.load_schema()
        model_ready = bool(model.is_fitted and scaler is not None and len(schema) == 114)

        if model_ready:
            return jsonify({
                "status": "healthy",
                "service_ready": True,
                "model_loaded": True,
                "model_id": "EXP-019",
                "timestamp": now.isoformat(),
            }), 200
        else:
            return jsonify({
                "status": "degraded",
                "service_ready": False,
                "model_loaded": False,
                "timestamp": now.isoformat(),
            }), 503
    except Exception as e:
        logger.warning(f"Health check detected degraded service: {e}")
        return jsonify({
            "status": "degraded",
            "service_ready": False,
            "model_loaded": False,
            "timestamp": now.isoformat(),
        }), 503


@api_bp.route("/current", methods=["GET"])
def get_current_aqi() -> Any:
    """Return the latest observed telemetry observation without model inference.

    Returns:
        200 OK JSON containing latest observation values, category, and freshness metadata.
    """
    try:
        predictor = get_predictor()
        obs = predictor.get_latest_observation()
        return jsonify(obs), 200
    except FileNotFoundError as e:
        raise ServiceUnavailable("Observation telemetry dataset is currently unavailable.") from e
    except ValidationError as e:
        raise BadRequest(str(e)) from e


@api_bp.route("/forecast", methods=["GET"])
def get_forecast() -> Any:
    """Generate 72-hour AQI forecast with empirical error intervals and extreme event alerts.

    Query parameters:
        force_refresh (bool, optional): If 'true', bypass prediction cache and recompute
                                        from the latest stored features. Default 'false'.

    Returns:
        200 OK JSON matching the PredictionResult contract with explicit data freshness.
    """
    force_refresh = parse_bool_arg(request.args.get("force_refresh"), default=False)

    try:
        predictor = get_predictor()
        result = predictor.predict_latest(use_cache=True, force_refresh=force_refresh)
        return jsonify(result), 200
    except FileNotFoundError as e:
        raise ServiceUnavailable("Feature dataset or model artifacts are unavailable for inference.") from e
    except ValidationError as e:
        raise BadRequest(str(e)) from e


@api_bp.route("/model/info", methods=["GET"])
def get_model_info() -> Any:
    """Return production model architecture specification, feature schema, and benchmark metrics."""
    try:
        predictor = get_predictor()
        metadata = predictor.get_model_metadata()
        return jsonify(metadata), 200
    except Exception as e:
        logger.error(f"Error retrieving model metadata: {e}")
        raise ServiceUnavailable("Model metadata is currently unavailable.") from e
