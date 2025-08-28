"""Logging utilities for safe path handling and PHI protection."""

import hashlib
import os
from pathlib import Path


def mask_path_for_log(file_path: str | Path) -> str:
    """Mask file paths for logging to prevent PHI exposure.

    Replaces full paths with hash + extension for privacy.
    In debug mode (BGBR_DEBUG=1), returns full path.

    Args:
        file_path: Path to mask

    Returns:
        Masked path like '.edf#a1b2c3' or full path in debug mode

    Examples:
        >>> mask_path_for_log('/data/patients/john_smith_123.edf')
        '.edf#a1b2c3d4'
        >>> mask_path_for_log('/tmp/test.txt')  # with BGBR_DEBUG=1
        '/tmp/test.txt'
    """
    # In debug mode, show full paths for development
    if os.getenv("BGBR_DEBUG", "").strip() == "1":
        return str(file_path)

    # Convert to Path object for easier handling
    path = Path(file_path) if not isinstance(file_path, Path) else file_path

    # If path doesn't exist or is empty, return as-is
    if not str(path).strip():
        return "<empty_path>"

    # Get file extension
    suffix = path.suffix or ""

    # Create hash of the full path (deterministic but private)
    path_hash = hashlib.sha256(str(path).encode()).hexdigest()[:8]

    # Return masked format: extension + short hash
    return f"{suffix}#{path_hash}" if suffix else f"file#{path_hash}"
