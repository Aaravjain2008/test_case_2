"""
app/__init__.py
Package-level logger configuration for the RAG system.
"""
import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
