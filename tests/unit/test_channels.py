"""Tests for channel validation and mapping - REAL BEHAVIORAL TESTS, NO MOCKING."""

import pytest

from brain_go_brrr.infra.data.channels import (
    CHANNEL_ALIASES,
    CHANNELS_10_20_FULL,
    CHANNELS_TUAB_19,
    CHANNELS_TUEV_20,
    map_channels_to_indices,
    validate_channels,
)


class TestChannelConstants:
    """Test channel constant definitions."""

    def test_tuab_has_19_channels(self):
        """Test TUAB has exactly 19 channels."""
        assert len(CHANNELS_TUAB_19) == 19
        # Should NOT have Fz
        assert "FZ" not in CHANNELS_TUAB_19
        # Should have other key channels
        assert "FP1" in CHANNELS_TUAB_19
        assert "C3" in CHANNELS_TUAB_19
        assert "O1" in CHANNELS_TUAB_19

    def test_tuev_has_20_channels(self):
        """Test TUEV has exactly 20 channels."""
        assert len(CHANNELS_TUEV_20) == 20
        # Should have Fz
        assert "FZ" in CHANNELS_TUEV_20
        # Should NOT have Fpz
        assert "FPZ" not in CHANNELS_TUEV_20
        # Should have Oz
        assert "OZ" in CHANNELS_TUEV_20

    def test_channel_order_preserved(self):
        """Test channel order follows standard montage (frontal to occipital)."""
        # TUAB should start with frontal, end with occipital
        assert CHANNELS_TUAB_19[0] == "FP1"
        assert CHANNELS_TUAB_19[-1] == "O2"

        # TUEV should also follow this pattern
        assert CHANNELS_TUEV_20[0] == "FP1"
        assert CHANNELS_TUEV_20[-1] == "OZ"

    def test_no_duplicate_channels(self):
        """Test no duplicate channels in definitions."""
        assert len(set(CHANNELS_TUAB_19)) == len(CHANNELS_TUAB_19)
        assert len(set(CHANNELS_TUEV_20)) == len(CHANNELS_TUEV_20)

    def test_channel_aliases_map_old_to_new(self):
        """Test channel aliases map old naming to modern naming."""
        assert CHANNEL_ALIASES["T3"] == "T7"
        assert CHANNEL_ALIASES["T4"] == "T8"
        assert CHANNEL_ALIASES["T5"] == "P7"
        assert CHANNEL_ALIASES["T6"] == "P8"


class TestValidateChannels:
    """Test validate_channels function behavior."""

    def test_validate_exact_match(self):
        """Test validation passes for exact channel match."""
        channels = CHANNELS_TUAB_19.copy()
        # Should not raise
        validate_channels(channels, CHANNELS_TUAB_19, "TUAB")

    def test_validate_with_aliasing(self):
        """Test validation handles old channel names via aliasing."""
        # Use old naming
        channels = [
            "FP1", "FP2", "F7", "F3", "F4", "F8",
            "T3",  # Old name for T7
            "C3", "CZ", "C4",
            "T4",  # Old name for T8
            "T5",  # Old name for P7
            "P3", "PZ", "P4",
            "T6",  # Old name for P8
            "O1", "OZ", "O2"
        ]
        # Should pass with aliasing
        validate_channels(channels, CHANNELS_TUAB_19, "TUAB")

    def test_validate_wrong_count_raises(self):
        """Test validation fails for wrong channel count."""
        channels = CHANNELS_TUAB_19[:10]  # Only 10 channels
        with pytest.raises(ValueError, match="requires exactly 19 channels, got 10"):
            validate_channels(channels, CHANNELS_TUAB_19, "TUAB")

    def test_validate_missing_channel_raises(self):
        """Test validation fails for missing required channel."""
        channels = CHANNELS_TUAB_19.copy()
        channels[5] = "WRONG"  # Replace F8 with invalid channel
        with pytest.raises(ValueError, match="Missing: \\['F8'\\]"):
            validate_channels(channels, CHANNELS_TUAB_19, "TUAB")

    def test_validate_extra_channel_raises(self):
        """Test validation fails for extra unexpected channel."""
        channels = CHANNELS_TUAB_19.copy()
        channels[0] = "FZ"  # TUAB shouldn't have FZ
        with pytest.raises(ValueError, match="Extra: \\['FZ'\\]"):
            validate_channels(channels, CHANNELS_TUAB_19, "TUAB")

    def test_validate_both_missing_and_extra(self):
        """Test error message includes both missing and extra channels."""
        # Make sure we have the right count but wrong channels
        channels = ["FZ"] + CHANNELS_TUAB_19[:-1]  # Replace O2 with FZ
        with pytest.raises(ValueError) as exc_info:
            validate_channels(channels, CHANNELS_TUAB_19, "TUAB")
        
        error_msg = str(exc_info.value)
        assert "Missing: ['O2']" in error_msg
        assert "Extra: ['FZ']" in error_msg

    def test_validate_case_sensitive(self):
        """Test validation is case-sensitive."""
        channels = [ch.lower() for ch in CHANNELS_TUAB_19]  # Lowercase
        with pytest.raises(ValueError, match="Missing:"):
            validate_channels(channels, CHANNELS_TUAB_19, "TUAB")

    def test_validate_eeg_prefix_aliasing(self):
        """Test validation handles EEG prefix variations."""
        channels = CHANNELS_TUAB_19.copy()
        # Replace some with EEG prefix versions
        idx_t7 = channels.index("T7")
        channels[idx_t7] = "EEG T3-REF"  # Should map to T7
        
        # Should pass with aliasing
        validate_channels(channels, CHANNELS_TUAB_19, "TUAB")


class TestMapChannelsToIndices:
    """Test map_channels_to_indices function behavior."""

    def test_map_identical_channels(self):
        """Test mapping when source and target are identical."""
        source = CHANNELS_TUAB_19
        target = CHANNELS_TUAB_19
        mapping = map_channels_to_indices(source, target)

        # Should be identity mapping
        for i in range(len(CHANNELS_TUAB_19)):
            assert mapping[i] == i

    def test_map_reordered_channels(self):
        """Test mapping when channels are reordered."""
        source = ["C3", "FP1", "O1", "F3"]  # Different order
        target = ["FP1", "F3", "C3", "O1"]  # Target order
        mapping = map_channels_to_indices(source, target)

        assert mapping[0] == 2  # C3 at source[0] -> target[2]
        assert mapping[1] == 0  # FP1 at source[1] -> target[0]
        assert mapping[2] == 3  # O1 at source[2] -> target[3]
        assert mapping[3] == 1  # F3 at source[3] -> target[1]

    def test_map_with_aliasing(self):
        """Test mapping handles channel aliasing."""
        source = ["T3", "T4", "T5", "T6"]  # Old naming
        target = ["T7", "T8", "P7", "P8"]  # Modern naming
        mapping = map_channels_to_indices(source, target)

        assert mapping[0] == 0  # T3->T7
        assert mapping[1] == 1  # T4->T8
        assert mapping[2] == 2  # T5->P7
        assert mapping[3] == 3  # T6->P8

    def test_map_missing_channel_raises(self):
        """Test mapping raises when required channel is missing."""
        source = ["FP1", "C3", "O1"]  # Missing F3
        target = ["FP1", "F3", "C3", "O1"]  # Requires F3
        
        with pytest.raises(ValueError, match="Required channel F3 not found"):
            map_channels_to_indices(source, target)

    def test_map_subset_of_source(self):
        """Test mapping when target is subset of source."""
        source = CHANNELS_TUAB_19  # 19 channels
        target = ["FP1", "C3", "O1"]  # Just 3 channels
        mapping = map_channels_to_indices(source, target)

        # Should map only the requested channels
        assert len(mapping) == 3
        assert mapping[0] == 0  # FP1
        assert mapping[7] == 1  # C3
        assert mapping[16] == 2  # O1

    def test_map_preserves_duplicates_in_source(self):
        """Test mapping handles duplicate channels in source."""
        source = ["FP1", "C3", "FP1", "O1"]  # FP1 appears twice
        target = ["FP1", "C3", "O1"]
        mapping = map_channels_to_indices(source, target)

        # Should map first occurrence
        assert mapping[0] == 0  # First FP1 -> target[0]
        assert mapping[1] == 1  # C3 -> target[1]
        # Note: second FP1 at index 2 won't be in mapping
        assert mapping[3] == 2  # O1 -> target[2]

    def test_map_empty_channels(self):
        """Test mapping with empty channel lists."""
        mapping = map_channels_to_indices([], [])
        assert mapping == {}

    def test_map_tuab_to_tuev(self):
        """Test realistic mapping from TUAB to TUEV-like configuration."""
        # TUAB doesn't have FZ, so this should fail
        source = CHANNELS_TUAB_19
        target = CHANNELS_TUEV_20
        
        with pytest.raises(ValueError, match="Required channel FZ not found"):
            map_channels_to_indices(source, target)


class TestChannelCompatibility:
    """Test compatibility between different channel configurations."""

    def test_tuab_tuev_difference(self):
        """Test the key difference between TUAB and TUEV."""
        tuab_set = set(CHANNELS_TUAB_19)
        tuev_set = set(CHANNELS_TUEV_20)

        # TUEV has FZ that TUAB doesn't
        assert "FZ" in tuev_set
        assert "FZ" not in tuab_set

        # Both have 19 common channels
        common = tuab_set & tuev_set
        assert len(common) == 19

    def test_all_channels_uppercase(self):
        """Test all channel names are uppercase (standard convention)."""
        for ch in CHANNELS_TUAB_19:
            assert ch == ch.upper()
        
        for ch in CHANNELS_TUEV_20:
            assert ch == ch.upper()

    def test_realistic_tuab_validation(self):
        """Test realistic TUAB data validation scenario."""
        # Simulate data from TUAB with old naming
        raw_channels = [
            "EEG FP1-REF", "EEG FP2-REF", "EEG F7-REF", "EEG F3-REF",
            "EEG F4-REF", "EEG F8-REF", "EEG T3-REF", "EEG C3-REF",
            "EEG CZ-REF", "EEG C4-REF", "EEG T4-REF", "EEG T5-REF",
            "EEG P3-REF", "EEG PZ-REF", "EEG P4-REF", "EEG T6-REF",
            "EEG O1-REF", "EEG OZ-REF", "EEG O2-REF"
        ]
        
        # Clean up EEG prefix
        cleaned = [ch.replace("EEG ", "").replace("-REF", "") for ch in raw_channels]
        
        # Should validate with aliasing
        validate_channels(cleaned, CHANNELS_TUAB_19, "TUAB")