"""Domain ports for abnormality detection - PURE CLEAN ARCHITECTURE.

Following Uncle Bob's Clean Architecture:
- Domain defines interfaces (ports)
- Infrastructure implements them (adapters)
- Application wires them together (composition root)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


@runtime_checkable
class EEGPreprocessorPort(Protocol):
    """Port for EEG preprocessing - domain defines what it needs."""

    def transform(self, raw: MneRaw) -> npt.NDArray[np.float32]:
        """Transform raw EEG to preprocessed array.

        Args:
            raw: Raw EEG data (using forward ref to avoid MNE dependency)

        Returns:
            Preprocessed EEG array (channels x samples)
        """
        ...


@runtime_checkable
class AbnormalityHeadPort(Protocol):
    """Port for abnormality classification head."""

    def predict_proba(self, X: npt.NDArray[np.float32]) -> float:  # noqa: N803
        """Predict abnormality probability.

        Args:
            X: Preprocessed EEG features

        Returns:
            Probability of abnormality (0-1)
        """
        ...


@runtime_checkable
class FeatureExtractorPort(Protocol):
    """Port for feature extraction from EEG."""

    def extract(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # noqa: N803
        """Extract features from preprocessed EEG.

        Args:
            X: Preprocessed EEG array

        Returns:
            Feature vector
        """
        ...


# P1 FIX: LoggerPort moved to domain/protocols/logger.py


# Forward reference to avoid MNE dependency in domain
class MneRaw(Protocol):
    """Protocol for MNE Raw object - domain doesn't depend on MNE."""

    @property
    def info(self) -> dict[str, Any]:
        """Info dict with sfreq, ch_names, etc."""
        ...

    @property
    def ch_names(self) -> list[str]:
        """Channel names."""
        ...

    @property
    def n_times(self) -> int:
        """Number of time samples."""
        ...

    def get_data(self) -> npt.NDArray[np.float64]:
        """Get data array."""
        ...

    def copy(self) -> MneRaw:
        """Copy the raw object."""
        ...
