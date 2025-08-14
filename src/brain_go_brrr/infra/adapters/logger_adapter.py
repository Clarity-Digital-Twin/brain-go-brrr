"""Infrastructure adapter for logging.

This adapter implements the domain port for logging,
wrapping the actual Python logging implementation.
"""

from __future__ import annotations

import logging

from brain_go_brrr.domain.abnormal.ports import LoggerPort


class PythonLoggerAdapter(LoggerPort):
    """Adapter wrapping Python logger to implement domain port."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the adapter with a Python logger.

        Args:
            logger: Python logger instance (or create default)
        """
        self._logger = logger or logging.getLogger("brain_go_brrr.domain")

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)
