"""Infrastructure adapter for EEGPT feature extraction.

This adapter implements the domain port for feature extraction,
wrapping the actual EEGPT model implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from brain_go_brrr.domain.abnormal.ports import FeatureExtractorPort
from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel  # noqa: TC001

if TYPE_CHECKING:
    import numpy.typing as npt


class EEGPTFeatureExtractorAdapter(FeatureExtractorPort):
    """Adapter wrapping EEGPT model to implement feature extraction port."""

    def __init__(self, model: EEGPTModel) -> None:
        """Initialize the adapter with an EEGPT model.

        Args:
            model: EEGPT model instance (already loaded)
        """
        self._model = model

        # Ensure model is loaded
        if not hasattr(self._model, "is_loaded") or not self._model.is_loaded:
            raise RuntimeError("EEGPT model must be loaded before creating adapter")

    def extract(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:  # noqa: N803
        """Extract features from preprocessed EEG.

        Implements the domain port interface.

        Args:
            X: Preprocessed EEG array (channels x samples)

        Returns:
            Feature vector as float32
        """
        # Create channel names for EEGPT (it expects them)
        n_channels = X.shape[0]
        channel_names = [f"CH{i + 1}" for i in range(n_channels)]

        # Use EEGPT model to extract features
        features = self._model.extract_features(X, channel_names)

        # Ensure float32 output as domain expects
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        return features.astype(np.float32, copy=False)
