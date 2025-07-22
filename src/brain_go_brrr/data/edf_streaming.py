"""Memory-efficient EDF streaming utilities."""

import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np
import numpy.typing as npt

try:
    import mne
    import mne.io

    HAS_MNE = True
except ImportError:
    HAS_MNE = False

logger = logging.getLogger(__name__)


class EDFStreamer:
    """Streaming interface for EDF files to handle large recordings efficiently."""

    def __init__(self, file_path: str | Path, chunk_duration: float = 30.0) -> None:
        """Initialize the EDF streamer.

        Args:
            file_path: Path to the EDF file
            chunk_duration: Duration of chunks for streaming in seconds
        """
        self.file_path = Path(file_path)
        self.chunk_duration = chunk_duration
        self._raw: mne.io.Raw | None = None
        self._sfreq: float | None = None
        self._duration: float | None = None
        self._n_channels: int | None = None
        # Also maintain the old names for compatibility
        self.raw: mne.io.Raw | None = self._raw
        self.sampling_rate: float | None = self._sfreq
        self.duration: float | None = self._duration
        self.n_channels: int | None = self._n_channels

    def __enter__(self) -> "EDFStreamer":
        """Enter context manager."""
        self.load_header()
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object | None
    ) -> None:
        """Exit context manager and clean up resources."""
        if self._raw is not None:
            self._raw.close()
            self._raw = None

    def load_header(self) -> None:
        """Load EDF header information without reading data."""
        if not HAS_MNE:
            raise ImportError("MNE-Python is required for EDF streaming")

        # Read header only
        self._raw = mne.io.read_raw_edf(self.file_path, preload=False)
        if self._raw is not None:
            self._sfreq = float(self._raw.info["sfreq"])
            self._duration = float(self._raw.times[-1])
            self._n_channels = len(self._raw.ch_names)
            # Update compatibility attributes
            self.raw = self._raw
            self.sampling_rate = self._sfreq
            self.duration = self._duration
            self.n_channels = self._n_channels

    def stream_chunks(self) -> Generator[tuple[npt.NDArray[np.float64], float], None, None]:
        """Stream data in chunks.

        Yields:
            Tuple of (data_chunk, start_time) where data_chunk is shape (n_channels, n_samples)
        """
        if self._raw is None:
            self.load_header()

        if self._sfreq is None or self._duration is None or self._raw is None:
            raise ValueError("File not loaded")

        chunk_samples = int(self.chunk_duration * self._sfreq)
        total_samples = int(self._duration * self._sfreq)

        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(start_sample + chunk_samples, total_samples)

            # Read data for this chunk
            data, _ = self._raw[:, start_sample:end_sample]
            start_time = start_sample / self._sfreq

            yield data, start_time
            start_sample = end_sample

    def process_in_windows(
        self, window_duration: float = 4.0, overlap: float = 0.0
    ) -> Generator[tuple[npt.NDArray[np.float64], float], None, None]:
        """Process data in analysis windows.

        Args:
            window_duration: Duration of each window in seconds
            overlap: Overlap fraction (0.0 to 1.0)

        Yields:
            Tuple of (data_window, start_time)
        """
        if overlap < 0 or overlap >= 1:
            raise ValueError("Overlap must be between 0 and 1")

        if self._raw is None:
            self.load_header()

        if self._sfreq is None or self._duration is None or self._raw is None:
            raise ValueError("File not loaded")

        step_duration = window_duration * (1 - overlap)
        window_samples = int(window_duration * self._sfreq)
        step_samples = int(step_duration * self._sfreq)
        total_samples = int(self._duration * self._sfreq)

        # Process all windows
        start_sample = 0
        while start_sample + window_samples <= total_samples:
            end_sample = start_sample + window_samples

            # Read data for this window
            data, _ = self._raw[:, start_sample:end_sample]
            start_time = start_sample / self._sfreq

            yield data, start_time
            start_sample += step_samples

    def stream_array(
        self, window_duration: float = 4.0, step_duration: float | None = None
    ) -> Generator[tuple[npt.NDArray[np.float64], int], None, None]:
        """Stream EEG data in windows.

        Args:
            window_duration: Duration of each window in seconds
            step_duration: Step size in seconds (defaults to window_duration)

        Yields:
            Tuple of (data_window, start_sample) where data_window is shape (n_channels, n_samples)
        """
        if self._raw is None:
            self.load_header()

        if self._sfreq is None:
            raise ValueError("Sampling rate not available")

        if step_duration is None:
            step_duration = window_duration

        window_samples = int(window_duration * self._sfreq)
        step_samples = int(step_duration * self._sfreq)

        if self._raw is None or self._duration is None:
            return

        # Convert duration to samples
        total_samples = int(self._duration * self._sfreq)

        # Stream windows
        start_sample = 0
        while start_sample + window_samples <= total_samples:
            end_sample = start_sample + window_samples

            # Read data for this window
            data, _ = self._raw[:, start_sample:end_sample]

            yield data, start_sample
            start_sample += step_samples

    def estimate_memory_usage(self, window_duration: float = 4.0) -> float:
        """Estimate memory usage for streaming in MB.

        Args:
            window_duration: Duration of window in seconds

        Returns:
            Estimated memory usage in MB
        """
        if self._n_channels is None or self._sfreq is None:
            self.load_header()

        if self._n_channels is None or self._sfreq is None:
            return 0.0

        # Calculate memory for one window
        window_samples = int(window_duration * self._sfreq)
        bytes_per_sample = 8  # float64
        window_memory_bytes = self._n_channels * window_samples * bytes_per_sample

        return window_memory_bytes / (1024 * 1024)  # Convert to MB

    def get_info(self) -> dict:
        """Get basic file information.

        Returns:
            Dictionary with file information
        """
        if self._raw is None:
            self.load_header()

        return {
            "file_path": str(self.file_path),
            "duration": self._duration,
            "sampling_rate": self._sfreq,
            "n_channels": self._n_channels,
            "channel_names": self._raw.ch_names if self._raw else [],
            "file_size_mb": self.file_path.stat().st_size / (1024 * 1024)
            if self.file_path.exists()
            else 0.0,
        }

    def get_file_info(self) -> dict:
        """Get basic file information.

        Returns:
            Dictionary with file information
        """
        return self.get_info()


def process_large_edf(file_path: str | Path, max_memory_mb: float = 500.0) -> dict:
    """Process EDF file with streaming decision based on memory.

    Args:
        file_path: Path to EDF file
        max_memory_mb: Maximum memory to use in MB

    Returns:
        Dictionary with processing results
    """
    # Estimate memory usage
    estimate = estimate_memory_usage(file_path, preload=True)

    # Decide whether to stream
    use_streaming = estimate["estimated_total_mb"] > max_memory_mb

    if use_streaming:
        # Process with streaming
        streamer = EDFStreamer(file_path)
        chunks = list(streamer.stream_chunks())
        return {
            "used_streaming": True,
            "chunks_processed": len(chunks),
            "memory_estimate_mb": estimate["estimated_total_mb"],
        }
    else:
        # Process without streaming - load full file
        if not HAS_MNE:
            raise ImportError("MNE-Python is required for EDF processing")

        # Load the entire file into memory
        raw = mne.io.read_raw_edf(file_path, preload=True)

        return {
            "used_streaming": False,
            "chunks_processed": 1,
            "memory_estimate_mb": estimate["estimated_total_mb"],
            "data_shape": (len(raw.ch_names), raw.n_times),
        }


def estimate_memory_usage(file_path: str | Path, preload: bool = True) -> dict:
    """Estimate memory usage for loading an EDF file.

    Args:
        file_path: Path to EDF file
        preload: Whether data would be preloaded

    Returns:
        Dictionary with memory estimates
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {
            "raw_data_mb": 0.0,
            "estimated_total_mb": 0.0,
            "duration_minutes": 0.0,
            "n_channels": 0,
            "sampling_rate": 0.0,
            "preload": preload,
        }

    try:
        streamer = EDFStreamer(file_path)
        info = streamer.get_file_info()

        if info["duration"] is None or info["sampling_rate"] is None or info["n_channels"] is None:
            return {
                "raw_data_mb": 0.0,
                "estimated_total_mb": 0.0,
                "duration_minutes": 0.0,
                "n_channels": 0,
                "sampling_rate": 0.0,
                "preload": preload,
            }

        # Calculate memory requirements
        total_samples = int(info["duration"] * info["sampling_rate"])
        bytes_per_sample = 8  # float64
        total_bytes = info["n_channels"] * total_samples * bytes_per_sample

        # Add overhead for MNE structures (~20%)
        overhead_factor = 1.2 if preload else 0.1
        estimated_bytes = total_bytes * overhead_factor

        return {
            "raw_data_mb": total_bytes / (1024 * 1024),
            "estimated_total_mb": estimated_bytes / (1024 * 1024),
            "duration_minutes": info["duration"] / 60,
            "n_channels": info["n_channels"],
            "sampling_rate": info["sampling_rate"],
            "preload": preload,
        }

    except Exception:
        # Return safe defaults on any error
        return {
            "raw_data_mb": 0.0,
            "estimated_total_mb": 10.0,  # Conservative estimate
            "duration_minutes": 0.0,
            "n_channels": 0,
            "sampling_rate": 0.0,
            "preload": preload,
        }


def decide_streaming(
    file_path: Path, max_memory_mb: float = 500.0
) -> dict[str, bool | int | float]:
    """Decide whether to use streaming based on memory requirements.

    This function estimates the memory needed to load an EDF file and determines
    whether streaming should be used based on a memory threshold.

    Args:
        file_path: Path to EDF file
        max_memory_mb: Maximum memory threshold in MB (default: 500MB)

    Returns:
        Dictionary containing:
            - used_streaming: Whether streaming is recommended
            - chunks_processed: Number of chunks needed if streaming
            - estimated_memory_mb: Estimated total memory usage
            - threshold_mb: The memory threshold used
            - chunk_duration_s: Duration of each chunk in seconds if streaming

    Examples:
        >>> result = decide_streaming(Path("large_file.edf"), max_memory_mb=100)
        >>> if result["used_streaming"]:
        ...     logger.info(f"Process in {result['chunks_processed']} chunks")
    """
    # Validate inputs
    if max_memory_mb <= 0:
        raise ValueError("max_memory_mb must be positive")

    # Get memory estimate
    estimate = estimate_memory_usage(file_path, preload=True)

    # Check if estimation was successful
    if estimate["estimated_total_mb"] == 0:
        logger.warning(f"Could not estimate memory for {file_path}, defaulting to streaming")
        return {
            "used_streaming": True,
            "chunks_processed": 1,
            "estimated_memory_mb": 0.0,
            "threshold_mb": max_memory_mb,
            "chunk_duration_s": 30.0,
        }

    # Decide based on memory threshold
    use_streaming = estimate["estimated_total_mb"] > max_memory_mb

    if use_streaming:
        # Calculate optimal chunk size
        # Use 80% of threshold to leave headroom for processing
        chunk_memory_mb = max_memory_mb * 0.8
        total_memory_mb = estimate["estimated_total_mb"]

        # Calculate chunks needed
        chunks_needed = max(1, int(np.ceil(total_memory_mb / chunk_memory_mb)))

        # Calculate chunk duration based on memory per second
        if estimate["duration_minutes"] > 0 and estimate["estimated_total_mb"] > 0:
            memory_per_second = estimate["estimated_total_mb"] / (estimate["duration_minutes"] * 60)
            chunk_duration_s = chunk_memory_mb / memory_per_second
            # Round to nearest 10 seconds for cleaner chunks
            chunk_duration_s = max(10.0, round(chunk_duration_s / 10) * 10)
        else:
            chunk_duration_s = 30.0  # Default chunk size
    else:
        chunks_needed = 1
        chunk_duration_s = (
            estimate["duration_minutes"] * 60 if estimate["duration_minutes"] > 0 else 0.0
        )

    return {
        "used_streaming": use_streaming,
        "chunks_processed": chunks_needed,
        "estimated_memory_mb": estimate["estimated_total_mb"],
        "threshold_mb": max_memory_mb,
        "chunk_duration_s": chunk_duration_s,
    }
