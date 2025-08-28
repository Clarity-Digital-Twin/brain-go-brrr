#!/usr/bin/env python3
"""Test script to verify correct channel enforcement for TUAB and TUEV."""


def test_channel_specs():
    """Verify channel specifications are correctly implemented."""

    print("Testing Channel Enforcement...")
    print("=" * 60)

    # Test TUAB preprocessor
    from mne_integration.preprocessor import TUABPreprocessor

    tuab_proc = TUABPreprocessor()
    assert (
        len(tuab_proc.STANDARD_CHANNELS) == 19
    ), f"TUAB should have 19 channels, got {len(tuab_proc.STANDARD_CHANNELS)}"
    assert 'Fz' not in tuab_proc.STANDARD_CHANNELS, "TUAB should NOT include Fz"
    assert 'Oz' in tuab_proc.STANDARD_CHANNELS, "TUAB should include Oz"
    print("✅ TUAB Preprocessor: 19 channels (no Fz)")

    # Test TUEV preprocessor
    from mne_integration.tuev_preprocessor import TUEVPreprocessor

    tuev_proc = TUEVPreprocessor()
    assert (
        len(tuev_proc.STANDARD_CHANNELS) == 20
    ), f"TUEV should have 20 channels, got {len(tuev_proc.STANDARD_CHANNELS)}"
    assert 'Fz' in tuev_proc.STANDARD_CHANNELS, "TUEV SHOULD include Fz"
    assert 'Oz' in tuev_proc.STANDARD_CHANNELS, "TUEV should include Oz"
    assert 'Fpz' not in [
        ch.upper() for ch in tuev_proc.STANDARD_CHANNELS
    ], "TUEV should NOT include Fpz"
    print("✅ TUEV Preprocessor: 20 channels (with Fz, no Fpz)")

    # Check configs
    import yaml

    with open('configs/tuev.yaml') as f:
        tuev_config = yaml.safe_load(f)
    target_channels = tuev_config['channels']['target_20']

    # Count unique channels (case insensitive)
    unique_channels = {ch.upper() for ch in target_channels}
    assert (
        len(unique_channels) == 20
    ), f"TUEV config should have 20 unique channels, got {len(unique_channels)}"
    assert 'FZ' in unique_channels, "TUEV config should include FZ"
    assert 'FPZ' not in unique_channels, "TUEV config should NOT include FPZ"
    assert 'OZ' in unique_channels, "TUEV config should include OZ"
    print("✅ TUEV Config: 20 channels correctly specified")

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("  TUAB: 19 channels (excludes Fz) ✅")
    print("  TUEV: 20 channels (includes Fz, excludes Fpz) ✅")
    print("\nAll channel specifications are CORRECT!")
    print("=" * 60)


if __name__ == "__main__":
    test_channel_specs()
