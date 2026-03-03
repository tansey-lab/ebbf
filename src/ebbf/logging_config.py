"""Centralized logging configuration for ebbf."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the ebbf package.

    Args:
        level: The logging level to use. Defaults to INFO.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    # Configure the root logger for the ebbf package
    logger = logging.getLogger("ebbf")
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
