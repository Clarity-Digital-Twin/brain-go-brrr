"""Modern logging setup with Rich formatting."""

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Path | None = None,
    rich_console: bool = True,
) -> logging.Logger:
    """Get a configured logger with Rich formatting.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        rich_console: Whether to use Rich console formatting

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers - return existing logger if already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    # Rich console handler
    if rich_console:
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        rich_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(rich_handler)
    else:
        # Standard console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
