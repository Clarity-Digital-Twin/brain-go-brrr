"""EEGPT feature dimension constants - SINGLE SOURCE OF TRUTH.

This module defines canonical constants for EEGPT feature dimensions.
These are used for documentation and testing purposes.

Note: The actual logic remains in utils/probe_utils.py:prepare_probe_features
to avoid circular dependencies and maintain separation of concerns.
"""

# EEGPT Large Model Feature Dimensions
EEGPT_SUMMARY_TOKENS = 4
"""Number of summary tokens output by EEGPT encoder."""

EEGPT_TOKEN_DIM = 512
"""Dimension of each EEGPT token embedding."""

EEGPT_PROBE_INPUT_DIM = EEGPT_SUMMARY_TOKENS * EEGPT_TOKEN_DIM  # 2048
"""Total dimension when all summary tokens are flattened for probe input."""

# Standard channel configurations
EEGPT_STANDARD_CHANNELS = 20
"""Standard number of EEG channels expected by EEGPT."""

TUAB_CHANNELS = 20
"""Number of channels in TUAB dataset (after mapping)."""

TUEV_CHANNELS = 23
"""Number of channels in TUEV dataset (before optional synthesis)."""

# Window specifications
EEGPT_WINDOW_SECONDS = 4.0
"""Standard window size in seconds for EEGPT processing."""

EEGPT_SAMPLING_RATE = 256
"""Target sampling rate in Hz for EEGPT."""

EEGPT_WINDOW_SAMPLES = int(EEGPT_WINDOW_SECONDS * EEGPT_SAMPLING_RATE)  # 1024
"""Number of samples in a standard EEGPT window."""
