"""Infrastructure adapter for flexible EEG preprocessing.

This adapter implements the domain port, allowing the domain to use
preprocessing without depending on the infrastructure implementation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from brain_go_brrr.domain.abnormal.ports import EEGPreprocessorPort, MneRaw
from brain_go_brrr.infra.preprocessing.flexible_preprocessor import (
    FlexibleEEGPreprocessor,
)


class FlexiblePreprocessorAdapter(EEGPreprocessorPort):
    """Adapter wrapping FlexibleEEGPreprocessor to implement domain port."""
    
    def __init__(
        self,
        target_sfreq: int = 256,
        lowpass_freq: float = 45.0,
        highpass_freq: float = 0.5,
        notch_freq: float = 50.0,
        **kwargs
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
            **kwargs
        )
    
    def transform(self, raw: MneRaw) -> npt.NDArray[np.float32]:
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