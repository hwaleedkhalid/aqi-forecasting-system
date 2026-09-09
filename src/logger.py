"""Pearls AQI Predictor - Logging Configuration.

Sets up a centralized logger with both console and file handlers.
Log format and level follow the standards defined in Rules.md §5.
"""

import logging
import sys
from pathlib import Path

from src.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOGS_DIR


def setup_logger(name: str = "pearls_aqi") -> logging.Logger:
    """Configure and return the application logger.

    Creates a logger with both console (stdout) and file handlers.
    The log file is stored in data/logs/ as specified in the architecture.

    Args:
        name: Name for the logger instance.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (data/logs/) - guarded for cloud container environments
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOGS_DIR / LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass

    return logger


# Module-level logger for convenience
logger = setup_logger()
