"""Unit tests for channel aliasing without requiring data."""

import pytest

from brain_go_brrr.infra.data.channels import CHANNEL_ALIASES, CHANNELS_TUAB_19


@pytest.mark.unit
@pytest.mark.synth
def test_channel_aliases_old_to_modern():
    """Test that old 10-20 names map to modern equivalents."""
    # Critical aliases for TUSZ compatibility
    assert CHANNEL_ALIASES["T3"] == "T7"
    assert CHANNEL_ALIASES["T4"] == "T8"
    assert CHANNEL_ALIASES["T5"] == "P7"
    assert CHANNEL_ALIASES["T6"] == "P8"
    
    # Also check uppercase variants
    assert CHANNEL_ALIASES["EEG T3-REF"] == "T7"
    assert CHANNEL_ALIASES["EEG T4-REF"] == "T8"


@pytest.mark.unit  
@pytest.mark.synth
def test_channels_tuab_19_uses_modern_naming():
    """Test that TUAB channel list uses modern T7/T8/P7/P8 names."""
    channels = CHANNELS_TUAB_19
    
    # Should have exactly 19 channels
    assert len(channels) == 19
    
    # Should use modern names
    assert "T7" in channels
    assert "T8" in channels
    assert "P7" in channels
    assert "P8" in channels
    
    # Should NOT have old names
    assert "T3" not in channels
    assert "T4" not in channels
    assert "T5" not in channels
    assert "T6" not in channels
    
    # Should have standard channels
    required = {"Fp1", "Fp2", "F7", "F3", "F4", "F8", "C3", "Cz", "C4", "O1", "O2"}
    for ch in required:
        assert ch in channels, f"Missing standard channel: {ch}"


@pytest.mark.unit
@pytest.mark.synth
def test_channel_aliases_comprehensive():
    """Test various alias formats are handled."""
    from brain_go_brrr.infra.data.tusz_detection_dataset import _standardize_channel_name
    
    # Test direct passthrough
    assert _standardize_channel_name("T7") == "T7"
    assert _standardize_channel_name("Fp1") == "Fp1"
    
    # Test aliasing
    assert _standardize_channel_name("T3") == "T7"
    assert _standardize_channel_name("T4") == "T8"
    
    # Test with EEG prefix (common in clinical files)
    assert _standardize_channel_name("EEG T3-REF") == "T7"
    assert _standardize_channel_name("EEG FP1-REF") == "Fp1"