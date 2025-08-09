"""Safe EDF file loading with proper error handling."""

import logging
from pathlib import Path
from typing import Any, cast

import mne

from brain_go_brrr._typing import MNERaw
from brain_go_brrr.core.exceptions import EdfLoadError

logger = logging.getLogger(__name__)


def validate_edf_path(file_path: Path | str) -> Path:
    """Validate EDF file path exists and is readable.
    
    Args:
        file_path: Path to the EDF file
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not .edf or .bdf
        PermissionError: If file is not readable
    """
    path = Path(file_path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")

    # Check if it's a file (not directory)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Check file extension
    if path.suffix.lower() not in [".edf", ".bdf"]:
        raise ValueError(f"File must be .edf or .bdf, got: {path.suffix}")

    # Check if readable
    try:
        with open(path, "rb") as f:
            # Try to read first byte to verify access
            f.read(1)
    except PermissionError:
        raise PermissionError(f"Cannot read file: {path}")
    except Exception as e:
        raise ValueError(f"Cannot access file: {e}")

    return path


def load_edf_safe(file_path: Path | str, **kwargs: Any) -> MNERaw:
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
        return cast("MNERaw", mne.io.read_raw_edf(file_path, **kwargs))
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
