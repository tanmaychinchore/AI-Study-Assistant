"""
Structured logging configuration for the RAG service.

Provides a consistent log format across all modules with timestamp,
level, module name, and message. Log level is controlled via the
LOG_LEVEL environment variable.
"""

import contextvars
import logging
import sys

from app.core.config import settings

# Thread-safe ContextVar to hold correlation IDs
request_id_var = contextvars.ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | [%(request_id)s] | %(name)-30s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RequestIDFilter(logging.Filter):
    """Logging filter to inject request_id from context variables into log records."""
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """Configure the root logger with a console handler and structured format."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on repeated calls
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Add Request ID filter to console handler
        console_handler.addFilter(RequestIDFilter())

        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(formatter)

        root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Usage:
        from app.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")
    """
    return logging.getLogger(name)
