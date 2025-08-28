"""Logging utilities for safe path handling and PHI protection."""

from __future__ import annotations

import hashlib
import os
import re
import typing as t
from pathlib import Path


def mask_path_for_log(file_path: str | Path) -> str:
    """Mask file paths for logging to prevent PHI exposure.

    Replaces full paths with hash + extension for privacy.
    In debug mode (BGBR_DEBUG=1), returns full path.

    Args:
        file_path: Path to mask

    Returns:
        Masked path like '.edf#a1b2c3' or full path in debug mode
    """
    # In debug mode, show full paths for development
    if os.getenv("BGBR_DEBUG", "").strip() == "1":
        return str(file_path)

    # Handle empty string explicitly
    if isinstance(file_path, str) and not file_path.strip():
        return "<empty_path>"

    # Convert to Path object for easier handling
    path = Path(file_path) if not isinstance(file_path, Path) else file_path

    s = str(path).strip()
    if not s:
        return "<empty_path>"

    # Get file extension
    suffix = path.suffix or ""

    # Create hash of the full path (deterministic but private)
    path_hash = hashlib.sha256(s.encode()).hexdigest()[:8]

    # Return masked format: extension + short hash
    return f"{suffix}#{path_hash}" if suffix else f"file#{path_hash}"


_PATH_EXTENSIONS = (
    "edf",
    "pt",
    "ckpt",
    "npz",
    "npy",
    "txt",
    "yaml",
    "yml",
    "json",
    "csv",
    "pdf",
    "md",
    "png",
    "jpg",
)

# Rough path-like pattern catcher: absolute or relative with separators and an extension
_PATH_PATTERN = re.compile(
    rf"(?P<path>(?:[A-Za-z]:\\\\|/|\\./|\.\./)[^\s\"']+\.({'|'.join(_PATH_EXTENSIONS)}))",
    re.IGNORECASE,
)


def _mask_paths_in_message(message: str) -> str:
    """Mask path-like substrings inside a log message string.

    Best-effort safety net in case callers forget to use mask_path_for_log.
    """
    if os.getenv("BGBR_DEBUG", "").strip() == "1":
        return message

    def repl(match: re.Match[str]) -> str:
        full = match.group("path")
        return mask_path_for_log(full)

    try:
        return _PATH_PATTERN.sub(repl, message)
    except Exception:
        # Never let logging crash the app
        return message


class PathMaskingFilter:
    """Logging filter that masks path-like substrings in log records."""

    def filter(self, record: t.Any) -> bool:  # logging.LogRecord type without import cycle
        try:
            # record.msg can be either a format string or formatted string.
            # Our code uses f-strings so it's already formatted.
            if isinstance(record.msg, str):
                record.msg = _mask_paths_in_message(record.msg)
            # If args exist for %-formatting, mask string args as a safety net
            if record.args:
                if isinstance(record.args, tuple):
                    record.args = tuple(
                        _mask_paths_in_message(a) if isinstance(a, str) else a for a in record.args
                    )
                elif isinstance(record.args, dict):
                    record.args = {
                        k: _mask_paths_in_message(v) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
        except Exception:
            # Never block logging; if masking fails, pass through
            pass
        return True


def add_path_masking(logger: t.Any) -> None:
    """Attach PathMaskingFilter to a logger and its handlers once."""
    try:
        # Avoid duplicate filters
        for f in logger.filters:
            if isinstance(f, PathMaskingFilter):
                break
        else:
            logger.addFilter(PathMaskingFilter())
        for h in getattr(logger, "handlers", []):
            for f in h.filters:
                if isinstance(f, PathMaskingFilter):
                    break
            else:
                h.addFilter(PathMaskingFilter())
    except Exception:
        # Never fail on logging setup
        pass
