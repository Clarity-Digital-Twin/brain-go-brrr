"""Infrastructure adapter for flexible EEG preprocessing.

This adapter implements the domain port, allowing the domain to use
preprocessing without depending on the infrastructure implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from brain_go_brrr._typing import MNERaw

from brain_go_brrr.domain.ports import PreprocessorPort
from brain_go_brrr.infra.preprocessing.flexible_preprocessor import (
    FlexibleEEGPreprocessor,
)


class FlexiblePreprocessorAdapter(PreprocessorPort):
    """Adapter wrapping FlexibleEEGPreprocessor to implement domain port."""

    def __init__(
        self,
        target_sfreq: int = 256,
        lowpass_freq: float = 45.0,
        highpass_freq: float = 0.5,
        notch_freq: float = 50.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter with preprocessing parameters.

        Args:
            target_sfreq: Target sampling frequency
            lowpass_freq: Low-pass filter frequency
            highpass_freq: High-pass filter frequency
            notch_freq: Notch filter frequency
            **kwargs: Additional parameters for FlexibleEEGPreprocessor
        """
        self._preprocessor = FlexibleEEGPreprocessor(
            target_sfreq=target_sfreq,
            lowpass_freq=lowpass_freq,
            highpass_freq=highpass_freq,
            notch_freq=notch_freq,
            **kwargs,
        )

    def preprocess(
        self,
        raw: MNERaw,
        bandpass: tuple[float, float] | None = None,
        notch: float | None = None,
    ) -> MNERaw:
        """Preprocess EEG data.

        Args:
            raw: Raw EEG data
            bandpass: Bandpass filter frequencies (low, high)
            notch: Notch filter frequency

        Returns:
            Preprocessed EEG data
        """
        # Update preprocessor parameters if provided
        if bandpass is not None:
            self._preprocessor.highpass_freq = bandpass[0]
            self._preprocessor.lowpass_freq = bandpass[1]
        if notch is not None:
            self._preprocessor.notch_freq = notch

        # Use the infrastructure preprocessor
        processed = self._preprocessor.preprocess(raw.copy())
        return processed

    def transform_to_array(self, raw: MNERaw) -> npt.NDArray[np.float32]:
        """Transform MNE Raw to numpy array.

        Args:
            raw: MNE Raw object

        Returns:
            Numpy array of EEG data
        """
        data = raw.get_data()
        return data.astype(np.float32, copy=False)

    def transform(self, raw: MNERaw) -> npt.NDArray[np.float32]:
        """Transform raw EEG to preprocessed array.

        Implements the domain port interface.

        Args:
            raw: Raw EEG data (MNE Raw object)

        Returns:
            Preprocessed EEG array as float32 (channels x samples)
        """
        # Use the infrastructure preprocessor
        processed = self._preprocessor.preprocess(raw.copy())

        # Extract data and ensure float32 as domain expects
        data = processed.get_data()
        return data.astype(np.float32, copy=False)
