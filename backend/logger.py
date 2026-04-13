"""Structured logging setup using loguru."""

from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Configure loguru for structured, production-grade logging."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        "costguard.log",
        rotation="50 MB",
        retention="7 days",
        compression="gz",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} — {message}",
        serialize=False,
    )


__all__ = ["logger", "configure_logging"]
