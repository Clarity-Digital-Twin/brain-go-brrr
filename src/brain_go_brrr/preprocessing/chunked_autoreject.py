"""Memory-efficient AutoReject implementation for large datasets.

Clean, simple chunked processing - no overengineering.
Just works with large datasets without OOM.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import mne

# Only import if available
try:
    from autoreject import AutoReject
    HAS_AUTOREJECT = True
except ImportError:
    HAS_AUTOREJECT = False
    AutoReject = None  # type: ignore

logger = logging.getLogger(__name__)


class ChunkedAutoRejectProcessor:
    """Process large datasets with AutoReject using chunking and caching.

    Simple approach:
    1. Fit AutoReject on a representative subset
    2. Cache the parameters
    3. Apply to full dataset in chunks
    """

    def __init__(
        self,
        cache_dir: str | Path = "autoreject_cache",
        chunk_size: int = 100,
        n_interpolate: list[int] | None = None,
        consensus: float = 0.1,
        random_state: int = 42
    ):
        """Initialize processor.

        Args:
            cache_dir: Directory for caching fitted parameters
            chunk_size: Number of files to process at once
            n_interpolate: AutoReject interpolation parameters
            consensus: AutoReject consensus parameter
            random_state: Random seed for reproducibility
        """
        # Use env var if available
        env_cache = os.environ.get("BGB_AR_CACHE_DIR")
        if env_cache:
            self.cache_dir = Path(env_cache)
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.n_interpolate = n_interpolate or [1, 4]
        self.consensus = consensus
        self.random_state = random_state
        self.ar_params: dict[str, Any] | None = None
        self.is_fitted = False

        if not HAS_AUTOREJECT:
            logger.warning("AutoReject not available - install with: pip install autoreject")

    def has_cached_params(self) -> bool:
        """Check if pre-fitted parameters exist."""
        param_file = self.cache_dir / "autoreject_params.pkl"
        return param_file.exists()

    def fit_on_subset(self, file_paths: list[Path], n_samples: int = 200) -> None:
        """Fit AutoReject on a representative subset.

        Args:
            file_paths: List of EDF file paths
            n_samples: Number of files to use for fitting
        """
        if not HAS_AUTOREJECT:
            logger.error("Cannot fit - AutoReject not installed")
            return

        logger.info(f"Fitting AutoReject on {n_samples} samples...")

        # Sample files
        sampled_files = self._stratified_sample(file_paths, n_samples)

        # Load and concatenate epochs
        all_epochs = []
        for i, file_path in enumerate(sampled_files):
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

                # Basic preprocessing
                raw.filter(0.5, 50.0, fir_design='firwin', verbose=False)

                # Create epochs
                epochs = mne.make_fixed_length_epochs(
                    raw, duration=10.0, overlap=5.0, preload=True, verbose=False
                )

                if len(epochs) > 0:
                    all_epochs.append(epochs)

                if (i + 1) % 10 == 0:
                    logger.info(f"Loaded {i + 1}/{len(sampled_files)} files")

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        if not all_epochs:
            raise ValueError("No valid files found for AutoReject fitting")

        # Concatenate
        combined_epochs = mne.concatenate_epochs(all_epochs)
        logger.info(f"Fitting on {len(combined_epochs)} total epochs")

        # Fit AutoReject
        ar = AutoReject(
            n_interpolate=self.n_interpolate,
            consensus=self.consensus,
            n_jobs=1,  # Single job for memory efficiency
            random_state=self.random_state,
            verbose=False
        )

        ar.fit(combined_epochs)

        # Save parameters
        self._save_parameters(ar)
        self.ar_params = self._extract_parameters(ar)
        self.is_fitted = True

        logger.info("AutoReject fitting completed and cached")

    def transform_raw(self, raw: mne.io.Raw, window_adapter: Any) -> mne.io.Raw:
        """Apply fitted AutoReject to raw data.

        Args:
            raw: Raw EEG data
            window_adapter: WindowEpochAdapter instance

        Returns:
            Cleaned raw data
        """
        if not HAS_AUTOREJECT:
            logger.warning("AutoReject not available - returning original data")
            return raw

        if not self.is_fitted:
            self._load_parameters()

        # Convert to epochs
        epochs = window_adapter.raw_to_windowed_epochs(raw)

        # Apply AutoReject
        epochs_clean = self._apply_autoreject(epochs)

        # Convert back
        raw_clean = window_adapter.epochs_to_continuous(epochs_clean, raw)

        return raw_clean

    def _apply_autoreject(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply AutoReject with cached parameters.

        Simple transform using pre-fitted thresholds.
        """
        ar = self._create_autoreject_from_params()

        # Transform
        epochs_clean = ar.transform(epochs)

        n_rejected = len(epochs) - len(epochs_clean)
        rejection_rate = n_rejected / len(epochs) if len(epochs) > 0 else 0

        logger.debug(
            f"AutoReject: {n_rejected}/{len(epochs)} epochs rejected "
            f"({rejection_rate:.1%})"
        )

        return epochs_clean

    def _create_autoreject_from_params(self) -> AutoReject:
        """Create AutoReject instance from cached parameters."""
        ar = AutoReject(
            n_interpolate=self.n_interpolate,
            consensus=self.consensus,
            n_jobs=1,
            random_state=self.random_state,
            verbose=False
        )

        # Set pre-computed parameters
        if self.ar_params is not None:
            # These are the fitted attributes AutoReject needs
            ar.threshes_ = self.ar_params['thresholds']
            ar.consensus_ = self.ar_params['consensus']
            ar.n_interpolate_ = self.ar_params['n_interpolate']
            if 'picks' in self.ar_params:
                ar.picks_ = self.ar_params['picks']

        return ar

    def _save_parameters(self, ar: AutoReject) -> None:
        """Save fitted parameters to disk."""
        params = self._extract_parameters(ar)
        param_file = self.cache_dir / "autoreject_params.pkl"

        with param_file.open('wb') as f:
            pickle.dump(params, f)

        logger.info(f"AutoReject parameters saved to {param_file}")

    def _load_parameters(self) -> None:
        """Load pre-fitted parameters from disk."""
        param_file = self.cache_dir / "autoreject_params.pkl"

        if not param_file.exists():
            raise ValueError(f"No cached parameters found at {param_file}")

        with param_file.open('rb') as f:
            self.ar_params = pickle.load(f)

        self.is_fitted = True
        logger.info("AutoReject parameters loaded from cache")

    def _extract_parameters(self, ar: Any) -> dict[str, Any]:
        """Extract fitted parameters from AutoReject instance."""
        return {
            'thresholds': getattr(ar, 'threshes_', None),
            'consensus': getattr(ar, 'consensus_', None),
            'n_interpolate': getattr(ar, 'n_interpolate_', None),
            'picks': getattr(ar, 'picks_', None)
        }

    def _stratified_sample(self, file_paths: list[Path], n_samples: int) -> list[Path]:
        """Sample files for fitting.

        Simple random sampling - could be enhanced with stratification
        by label if needed.
        """
        import random

        random.seed(self.random_state)
        n_samples = min(n_samples, len(file_paths))

        return random.sample(file_paths, n_samples)
