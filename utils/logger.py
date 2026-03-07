"""
utils/logger.py
─────────────────────────────────────────────────────────────────
Centralized Loguru logger configuration.
All modules call `get_logger(__name__)` to get a named logger.
"""

import sys
from pathlib import Path
from loguru import logger as _logger


_configured = False


def setup_logging(log_level: str = "INFO", log_dir: str = "data/logs",
                  debug: bool = False) -> None:
    global _configured
    if _configured:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _logger.remove()  # Remove default handler

    # Console handler — human-readable
    fmt_console = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )
    _logger.add(sys.stderr, format=fmt_console, level=log_level,
                colorize=True, enqueue=True)

    # File handler — structured, rotating
    _logger.add(
        f"{log_dir}/aura_{{time:YYYY-MM-DD}}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} — {message}",
        level="DEBUG" if debug else log_level,
        rotation="100 MB",
        retention="14 days",
        compression="zip",
        enqueue=True
    )

    _configured = True


def get_logger(name: str):
    """Return a Loguru logger bound to the given module name."""
    return _logger.bind(name=name)
