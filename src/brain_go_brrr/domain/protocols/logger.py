"""Unified LoggerPort protocol for domain layer.

P1 FIX: Single source of truth for LoggerPort with flexible signature.
Replaces both domain/ports/base.py and domain/abnormal/ports.py versions.
"""

from typing import Any, Protocol


class LoggerPort(Protocol):
    """Unified logger interface for domain layer.

    Uses flexible signature (*args, **kwargs) to support both simple
    string messages and complex logging with extra context.
    """

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message with optional formatting."""
        ...

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info message with optional formatting."""
        ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message with optional formatting."""
        ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log error message with optional formatting."""
        ...
