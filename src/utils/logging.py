"""Lightweight logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "lhdl", level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger.

    Idempotent: repeated calls with the same name do not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)
    logger.propagate = False

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(handler)
    return logger
