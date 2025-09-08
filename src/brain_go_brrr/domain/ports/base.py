"""Domain ports (interfaces) for dependency inversion.

Following Clean Architecture: Domain defines interfaces, outer layers implement them.
This allows the domain to remain pure without any dependencies on infrastructure or application layers.
"""

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np
import numpy.typing as npt

from brain_go_brrr._typing import MNERaw


# P1 FIX: LoggerPort moved to domain/protocols/logger.py


class EEGModelPort(ABC):
    """Port for EEG feature extraction models."""

    @abstractmethod
    def extract_features(
        self, eeg_data: npt.NDArray[np.float32], sampling_rate: int = 256
    ) -> npt.NDArray[np.float32]:
        """Extract features from EEG data.

        Args:
            eeg_data: EEG data array (channels x samples)
            sampling_rate: Sampling rate in Hz

        Returns:
            Feature vector
        """
        pass

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        pass


class PreprocessorPort(ABC):
    """Port for EEG preprocessing."""

    @abstractmethod
    def preprocess(
        self, raw: MNERaw, bandpass: tuple[float, float] | None = None, notch: float | None = None
    ) -> MNERaw:
        """Preprocess EEG data.

        Args:
            raw: Raw EEG data
            bandpass: Bandpass filter frequencies (low, high)
            notch: Notch filter frequency

        Returns:
            Preprocessed EEG data
        """
        pass

    @abstractmethod
    def transform_to_array(self, raw: MNERaw) -> npt.NDArray[np.float32]:
        """Transform MNE Raw to numpy array.

        Args:
            raw: MNE Raw object

        Returns:
            Numpy array (channels x samples)
        """
        pass


class ConfigurationPort(Protocol):
    """Port for configuration access."""

    @property
    def model_path(self) -> str:
        """Path to model checkpoint."""
        ...

    @property
    def sampling_rate(self) -> int:
        """Target sampling rate."""
        ...

    @property
    def window_size(self) -> float:
        """Window size in seconds."""
        ...

    @property
    def bandpass_low(self) -> float:
        """Low frequency for bandpass filter."""
        ...

    @property
    def bandpass_high(self) -> float:
        """High frequency for bandpass filter."""
        ...


class AbnormalityConfigPort(Protocol):
    """Port for abnormality detection configuration."""

    @property
    def confidence_threshold(self) -> float:
        """Confidence threshold for abnormality detection."""
        ...

    @property
    def min_confidence(self) -> float:
        """Minimum confidence for valid prediction."""
        ...

    @property
    def channels(self) -> list[str]:
        """Required EEG channels."""
        ...
