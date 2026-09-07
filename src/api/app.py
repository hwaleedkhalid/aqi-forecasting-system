"""Pearls AQI Predictor - Flask Application Factory.

Initializes Flask application, registers CORS policies, mounts the /api blueprint,
and establishes centralized error handling.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, HTTPException, NotFound, ServiceUnavailable

from src.api.routes import api_bp
from src.logger import logger


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    """Application factory creating configured Flask application instance.

    Args:
        test_config: Optional dictionary of configuration overrides for testing.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)

    # Default configuration
    app.config.from_mapping(
        JSON_SORT_KEYS=False,
        CORS_ORIGINS=os.environ.get(
            "CORS_ORIGINS",
            "http://localhost:8501,http://127.0.0.1:8501",
        ).split(","),
    )

    if test_config:
        app.config.update(test_config)

    # Configure CORS for /api/* endpoints
    allowed_origins = [orig.strip() for orig in app.config.get("CORS_ORIGINS", ["*"])]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}})
    logger.info(f"Initialized Flask API with CORS origins: {allowed_origins}")

    # Register API blueprint
    app.register_blueprint(api_bp)

    # Register centralized error handlers
    @app.errorhandler(BadRequest)
    def handle_bad_request(e: BadRequest) -> Any:
        now = datetime.now(timezone.utc)
        return jsonify({
            "error": str(e.description),
            "status": 400,
            "timestamp": now.isoformat(),
        }), 400

    @app.errorhandler(NotFound)
    def handle_not_found(e: NotFound) -> Any:
        now = datetime.now(timezone.utc)
        return jsonify({
            "error": "Endpoint not found",
            "status": 404,
            "timestamp": now.isoformat(),
        }), 404

    @app.errorhandler(ServiceUnavailable)
    def handle_service_unavailable(e: ServiceUnavailable) -> Any:
        now = datetime.now(timezone.utc)
        return jsonify({
            "error": str(e.description),
            "status": 503,
            "timestamp": now.isoformat(),
        }), 503

    @app.errorhandler(Exception)
    def handle_unexpected_error(e: Exception) -> Any:
        # If it's already an HTTPException, handle accordingly
        if isinstance(e, HTTPException):
            now = datetime.now(timezone.utc)
            return jsonify({
                "error": str(e.description),
                "status": e.code or 500,
                "timestamp": now.isoformat(),
            }), e.code or 500

        # Log full exception details and stack trace server-side only
        logger.exception("Unhandled internal server error encountered during request processing")

        # Return generic message to client without leaking filesystem paths or exception traces
        now = datetime.now(timezone.utc)
        return jsonify({
            "error": "Internal server error",
            "status": 500,
            "timestamp": now.isoformat(),
        }), 500

    return app


# Canonical module-level WSGI application instance
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

