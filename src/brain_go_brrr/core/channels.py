"""Channel mapping and validation for EEG recordings.

Handles conversion from old to modern naming conventions.
"""

from dataclasses import dataclass


class ChannelMapper:
    """Maps old channel naming conventions to modern standards."""

    # Mapping from old to modern naming
    OLD_TO_MODERN = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
        # Handle uppercase variations
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "OZ": "Oz",
    }

    def standardize_channel_names(self, channels: list[str]) -> list[str]:
        """Convert channel names to modern convention.

        Args:
            channels: List of channel names in any convention

        Returns:
            List of standardized channel names
        """
        standardized = []

        for channel in channels:
            # First check if it needs old->modern mapping
            upper = channel.upper()
            if upper in self.OLD_TO_MODERN:
                standardized.append(self.OLD_TO_MODERN[upper])
            else:
                # Just normalize case for standard channels
                if len(channel) >= 2:
                    # First letter uppercase, rest lowercase except numbers
                    if channel[0].upper() == "F" and channel[1].upper() == "P":
                        standardized.append("Fp" + channel[2:])
                    else:
                        # Standard case: First uppercase, then lowercase, preserve numbers
                        result = channel[0].upper()
                        for char in channel[1:]:
                            if char.isdigit():
                                result += char
                            else:
                                result += char.lower()
                        standardized.append(result)
                else:
                    standardized.append(channel.upper())

        return standardized


class ChannelValidator:
    """Validates channel availability for EEGPT requirements."""

    # Required channels for EEGPT (19 channels)
    REQUIRED_CHANNELS = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
    ]

    def validate_channels(self, channels: list[str]) -> tuple[bool, list[str]]:
        """Check if recording has minimum required channels.

        Args:
            channels: List of available channel names

        Returns:
            Tuple of (is_valid, missing_channels)
        """
        channel_set = set(channels)
        missing = []

        for required in self.REQUIRED_CHANNELS:
            if required not in channel_set:
                missing.append(required)

        is_valid = len(missing) == 0
        return is_valid, missing

    def get_standard_channel_indices(self, channels: list[str]) -> dict[str, int]:
        """Get indices of standard channels in the input list.

        Args:
            channels: List of channel names

        Returns:
            Dict mapping standard channel names to their indices
        """
        indices = {}
        for i, channel in enumerate(channels):
            if channel in self.REQUIRED_CHANNELS:
                indices[channel] = i
        return indices


class ChannelSelector:
    """Selects and reorders channels to standard configuration."""

    # Canonical channel order for EEGPT
    CHANNEL_ORDER = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "O1",
        "O2",
    ]

    def select_standard_channels(self, channels: list[str]) -> tuple[list[str], list[int]]:
        """Select standard channels in canonical order.

        Args:
            channels: List of available channel names

        Returns:
            Tuple of (selected_channels, original_indices)
        """
        channel_to_index = {ch: i for i, ch in enumerate(channels)}

        selected = []
        indices = []

        for standard_ch in self.CHANNEL_ORDER:
            if standard_ch in channel_to_index:
                selected.append(standard_ch)
                indices.append(channel_to_index[standard_ch])

        return selected, indices


@dataclass
class ChannelProcessingResult:
    """Result of channel processing pipeline."""

    is_valid: bool
    standardized_names: list[str]
    selected_indices: list[int]
    missing_channels: list[str]
    selected_channels: list[str]


class ChannelProcessor:
    """Complete channel processing pipeline."""

    def __init__(self) -> None:
        """Initialize channel processor with mapper, validator, and selector."""
        self.mapper = ChannelMapper()
        self.validator = ChannelValidator()
        self.selector = ChannelSelector()

    def process_channels(self, raw_channels: list[str]) -> ChannelProcessingResult:
        """Process channels through complete pipeline.

        Args:
            raw_channels: Raw channel names from EEG file

        Returns:
            ChannelProcessingResult with all processing details
        """
        # Step 1: Standardize names
        standardized = self.mapper.standardize_channel_names(raw_channels)

        # Step 2: Validate
        is_valid, missing = self.validator.validate_channels(standardized)

        # Step 3: Select and reorder
        selected, indices = self.selector.select_standard_channels(standardized)

        return ChannelProcessingResult(
            is_valid=is_valid,
            standardized_names=standardized,
            selected_indices=indices,
            missing_channels=missing,
            selected_channels=selected,
        )
