"""Infrastructure adapter for AutoReject.

This adapter implements the domain port for artifact rejection,
allowing the domain to use AutoReject without depending on it.
"""

import logging
import sys
from pathlib import Path
from typing import Any

from brain_go_brrr._typing import MNEEpochs

# No sys.path hacks - use proper imports

try:
    from autoreject import AutoReject

    HAS_AUTOREJECT = True
except ImportError:
    logging.warning("autoreject not available. Install with: pip install autoreject")
    HAS_AUTOREJECT = False
    AutoReject = None


class AutoRejectAdapter:
    """Adapter for AutoReject to implement domain port."""

    def __init__(
        self,
        n_interpolate: list[int] | None = None,
        n_jobs: int = 1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        """Initialize AutoReject adapter.

        Args:
            n_interpolate: Number of channels to interpolate
            n_jobs: Number of parallel jobs
            random_state: Random seed
            verbose: Verbosity level
        """
        if not HAS_AUTOREJECT or AutoReject is None:
            self.autoreject = None
            logging.warning("AutoReject not available - will use basic rejection")
        else:
            self.autoreject = AutoReject(
                n_interpolate=n_interpolate or [1, 4, 8, 16],
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
            )

    def fit_transform(self, epochs: MNEEpochs) -> tuple[MNEEpochs, dict[str, Any]]:
        """Fit and transform epochs with rejection/interpolation.

        Args:
            epochs: Input epochs

        Returns:
            Tuple of (cleaned epochs, rejection info dict)
        """
        if self.autoreject is None:
            # Basic rejection without AutoReject
            return self._basic_rejection(epochs)

        # Use AutoReject
        epochs_clean = self.autoreject.fit_transform(epochs)

        # Extract rejection info - handle API changes
        rejection_info = {
            "reject_log": getattr(
                self.autoreject,
                "reject_log",
                getattr(self.autoreject, "get_reject_log", lambda _: None)(epochs_clean)
                if hasattr(self.autoreject, "get_reject_log")
                else None,
            ),
            "thresholds": getattr(self.autoreject, "thresholds_", {}),
            "interpolated": getattr(self.autoreject, "dots", {}),
        }

        return epochs_clean, rejection_info

    def _basic_rejection(self, epochs: MNEEpochs) -> tuple[MNEEpochs, dict[str, Any]]:
        """Basic artifact rejection without AutoReject.

        Simple threshold-based rejection as fallback.
        """
        import numpy as np

        data = epochs.get_data()
        reject_log = []

        for epoch in data:
            # Check for high amplitude (100 µV threshold)
            if np.max(np.abs(epoch)) > 100e-6 or np.std(epoch) < 1e-6:
                reject_log.append(True)
            else:
                reject_log.append(False)

        # Drop bad epochs
        good_epochs = [i for i, rejected in enumerate(reject_log) if not rejected]
        if good_epochs:
            # Create selection array for MNE
            selection = np.array(good_epochs)
            epochs_clean = epochs.copy()
            epochs_clean = epochs_clean[selection]  # type: ignore[assignment, index]
        else:
            epochs_clean = epochs.copy()  # Return copy if all bad

        rejection_info = {
            "reject_log": reject_log,
            "method": "basic_threshold",
            "n_rejected": sum(reject_log),
        }

        return epochs_clean, rejection_info
