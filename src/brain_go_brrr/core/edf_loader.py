"""Safe EDF file loading with proper error handling."""

import logging
from pathlib import Path
from typing import Any

import mne

from brain_go_brrr.core.exceptions import EdfLoadError

logger = logging.getLogger(__name__)


def load_edf_safe(file_path: Path | str, **kwargs: Any) -> mne.io.Raw:
    """Load EDF file with proper error translation.

    This wrapper provides consistent error handling for EDF loading,
    translating various MNE errors into a single EdfLoadError type.

    Args:
        file_path: Path to the EDF file
        **kwargs: Additional arguments passed to mne.io.read_raw_edf

    Returns:
        Loaded Raw object

    Raises:
        EdfLoadError: If the file cannot be loaded for any reason
    """
    try:
        return mne.io.read_raw_edf(file_path, **kwargs)
    except FileNotFoundError as e:
        raise EdfLoadError(f"EDF file not found: {file_path}") from e
    except ValueError as e:
        # MNE raises ValueError for corrupt/invalid EDF files
        raise EdfLoadError(f"Invalid EDF file format: {e}") from e
    except MemoryError as e:
        raise EdfLoadError(f"Insufficient memory to load EDF file: {e}") from e
    except OSError as e:
        # Catch file system errors (permissions, disk full, etc.)
        raise EdfLoadError(f"File system error loading EDF: {e}") from e
    except Exception as e:
        # Catch any other MNE-specific exceptions
        if "edf" in str(type(e).__name__).lower():
            raise EdfLoadError(f"EDF-specific error: {e}") from e
        # Re-raise unexpected errors
        raise
