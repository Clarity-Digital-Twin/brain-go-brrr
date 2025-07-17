"""Modern logging setup with Rich formatting."""

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
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
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # Rich console handler
    if rich_console:
        console = Console(stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    else:
        # Standard console handler
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
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