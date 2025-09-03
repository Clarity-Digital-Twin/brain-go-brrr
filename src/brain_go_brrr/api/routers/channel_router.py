"""Channel routing service for EEG analysis.

Routes EEG data to appropriate analysis methods based on channel count:
- <19 channels: Route to YASA (works with any channel count)
- >=19 channels: Route to EEGPT (requires 19+ channels)

Technical debt fix: Removes unnecessary 19-channel restriction.
YASA achieves 85%+ accuracy with just 1 central EEG channel.
"""

import logging
from typing import Any, Literal

import mne

logger = logging.getLogger(__name__)


class ChannelRouter:
    """Routes EEG data to appropriate analysis method based on channel availability."""

    # Minimum channels for each method
    MIN_CHANNELS_YASA = 1  # YASA works with ANY channel count
    MIN_CHANNELS_EEGPT = 19  # EEGPT requires standard 10-20 montage

    @classmethod
    def determine_analysis_method(
        cls, raw: mne.io.Raw, requested_method: Literal["auto", "yasa", "eegpt"] = "auto"
    ) -> tuple[str, dict[str, Any]]:
        """Determine best analysis method based on channel count.

        Args:
            raw: MNE Raw object with EEG data
            requested_method: User preference (auto selects best)

        Returns:
            Tuple of (method_name, metadata)

        Raises:
            ValueError: If requested method incompatible with data
        """
        n_channels = len(raw.ch_names)
        channel_names = raw.ch_names

        # Check central channels for YASA preference
        central_channels = ["C3", "C4", "Cz"]
        has_central = any(ch in channel_names for ch in central_channels)

        metadata = {
            "n_channels": n_channels,
            "has_central_channels": has_central,
            "channel_names": channel_names[:10],  # First 10 for logging
        }

        if requested_method == "eegpt":
            if n_channels < cls.MIN_CHANNELS_EEGPT:
                raise ValueError(
                    f"EEGPT requires at least {cls.MIN_CHANNELS_EEGPT} channels, "
                    f"found {n_channels}. Use YASA instead or set method='auto'."
                )
            logger.info(f"Using EEGPT with {n_channels} channels")
            return "eegpt", metadata

        elif requested_method == "yasa":
            if n_channels < cls.MIN_CHANNELS_YASA:
                raise ValueError(
                    f"YASA requires at least {cls.MIN_CHANNELS_YASA} channel, found {n_channels}"
                )
            logger.info(f"Using YASA with {n_channels} channels")
            return "yasa", metadata

        else:  # auto mode
            # Route based on channel availability
            if n_channels >= cls.MIN_CHANNELS_EEGPT:
                # Prefer EEGPT for full montage
                logger.info(f"Auto-routing to EEGPT ({n_channels} channels available)")
                return "eegpt", metadata
            else:
                # Use YASA for limited channels
                logger.info(
                    f"Auto-routing to YASA ({n_channels} channels, "
                    f"EEGPT needs {cls.MIN_CHANNELS_EEGPT})"
                )
                metadata["routing_reason"] = "insufficient_channels_for_eegpt"
                return "yasa", metadata

    @classmethod
    def validate_for_sleep_analysis(cls, raw: mne.io.Raw) -> tuple[bool, str]:
        """Validate EEG data for sleep analysis.

        Args:
            raw: MNE Raw object

        Returns:
            Tuple of (is_valid, message)
        """
        n_channels = len(raw.ch_names)

        # Check minimum requirements
        if n_channels < cls.MIN_CHANNELS_YASA:
            return False, f"Need at least {cls.MIN_CHANNELS_YASA} channel for analysis"

        # Check sampling rate
        sfreq = raw.info["sfreq"]
        if sfreq < 50:
            return False, f"Sampling rate too low ({sfreq}Hz), need at least 50Hz"

        # Check duration (need at least 30 seconds for one epoch)
        duration = raw.times[-1]
        if duration < 30:
            return False, f"Recording too short ({duration:.1f}s), need at least 30s"

        return True, "Data valid for sleep analysis"

    @classmethod
    def get_method_info(cls) -> dict[str, dict[str, Any]]:
        """Get information about available analysis methods."""
        return {
            "yasa": {
                "name": "YASA Sleep Staging",
                "min_channels": cls.MIN_CHANNELS_YASA,
                "optimal_channels": ["C3", "C4"],
                "accuracy": "85-87%",
                "description": "Works with any channel count, prefers central channels",
            },
            "eegpt": {
                "name": "EEGPT with Linear Probe",
                "min_channels": cls.MIN_CHANNELS_EEGPT,
                "required_channels": "Standard 10-20 montage",
                "accuracy": "85-90%",
                "description": "Deep learning model requiring full montage",
            },
        }
