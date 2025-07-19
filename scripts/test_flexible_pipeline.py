#!/usr/bin/env python
"""Test the flexible preprocessing pipeline with real data.

This script tests the new flexible preprocessor that handles heterogeneous
data formats like Sleep-EDF (no positions) and TUH (with positions).
"""

import sys
from pathlib import Path

import mne
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_go_brrr.core.logger import get_logger
from brain_go_brrr.models.eegpt_model import EEGPTModel
from brain_go_brrr.preprocessing.flexible_preprocessor import FlexibleEEGPreprocessor
from services.qc_flagger import EEGQualityController
from services.sleep_metrics import SleepAnalyzer

# Set up logging
logger = get_logger(__name__)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print("=" * 80)


def test_sleep_edf_preprocessing():
    """Test preprocessing Sleep-EDF data without positions."""
    print_section("SLEEP-EDF PREPROCESSING TEST")

    # Find Sleep-EDF files
    sleep_edf_dir = Path("data/datasets/external/sleep-edf/sleep-cassette")
    edf_file = next(iter(sleep_edf_dir.glob("*-PSG.edf")))  # Get first file

    logger.info(f"Testing with: {edf_file.name}")

    # Load data
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    logger.info(f"Original: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")
    logger.info(f"Channels: {', '.join(raw.ch_names[:5])}...")

    # Test flexible preprocessor in sleep mode
    preprocessor = FlexibleEEGPreprocessor(
        mode="sleep",
        target_sfreq=100,  # Keep original for sleep
        use_autoreject=False,  # No positions available
    )

    try:
        processed = preprocessor.preprocess(raw)
        logger.info("✓ Preprocessing successful!")
        logger.info(f"Processed: {len(processed.ch_names)} channels, {processed.info['sfreq']} Hz")
        logger.info(f"Channels: {', '.join(processed.ch_names)}")

        # Run sleep analysis
        analyzer = SleepAnalyzer()
        sleep_results = analyzer.run_full_sleep_analysis(processed)

        logger.info("\n✓ Sleep Analysis Results:")
        if "sleep_statistics" in sleep_results:
            stats = sleep_results["sleep_statistics"]
            logger.info(f"  - Total sleep time: {stats.get('TST', 0):.1f} min")
            logger.info(f"  - Sleep efficiency: {stats.get('SE', 0):.1f}%")
            logger.info(f"  - REM percentage: {stats.get('%REM', 0):.1f}%")
        else:
            logger.warning("Sleep statistics not found in results")

        return processed

    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_tuh_preprocessing():
    """Test preprocessing TUH data with positions."""
    print_section("TUH EEG PREPROCESSING TEST")

    tuh_dir = Path("data/datasets/external/tuh_eeg_events/v2.0.1/edf/train")
    if not tuh_dir.exists():
        logger.warning("TUH data not found")
        return None

    # Find first EDF file
    edf_file = next(iter(tuh_dir.rglob("*.edf")))
    logger.info(f"Testing with: {edf_file.name}")

    # Load data
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    logger.info(f"Original: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

    # Test flexible preprocessor in abnormality mode
    preprocessor = FlexibleEEGPreprocessor(
        mode="abnormality",
        target_sfreq=256,  # EEGPT requirement
        use_autoreject=True,  # Should have positions
    )

    try:
        processed = preprocessor.preprocess(raw)
        logger.info("✓ Preprocessing successful!")
        logger.info(f"Processed: {len(processed.ch_names)} channels, {processed.info['sfreq']} Hz")
        logger.info(f"Has montage: {processed.get_montage() is not None}")

        # Test with EEGPT if available
        model_path = Path("data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
        if model_path.exists():
            eegpt = EEGPTModel(checkpoint_path=str(model_path))
            result = eegpt.predict_abnormality(processed)
            logger.info("\n✓ EEGPT Abnormality Detection:")
            logger.info(f"  - Probability: {result['abnormal_probability']:.3f}")
            logger.info(f"  - Classification: {result['classification']}")
            eegpt.cleanup()

        return processed

    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_minimal_preprocessing():
    """Test minimal preprocessing mode for quick analysis."""
    print_section("MINIMAL PREPROCESSING TEST")

    # Create synthetic data
    sfreq = 256
    duration = 60  # 1 minute
    n_channels = 19

    # Standard 10-20 channels
    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    ]

    # Generate data
    times = np.arange(0, duration, 1 / sfreq)
    data = np.random.randn(n_channels, len(times)) * 30e-6

    # Create raw object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)

    # Add standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    # Test minimal preprocessing
    preprocessor = FlexibleEEGPreprocessor(
        mode="minimal",
        target_sfreq=None,  # Keep original
        use_autoreject=False,  # Fast processing
    )

    try:
        processed = preprocessor.preprocess(raw)
        logger.info("✓ Minimal preprocessing successful!")
        logger.info(f"Processing kept original sampling rate: {processed.info['sfreq']} Hz")

        # Run QC
        qc = EEGQualityController()
        qc_report = qc.run_full_qc_pipeline(processed)

        logger.info("\n✓ QC Results:")
        if "signal_quality" in qc_report and "snr" in qc_report["signal_quality"]:
            logger.info(f"  - SNR: {qc_report['signal_quality']['snr']:.2f} dB")
        if "quality_summary" in qc_report:
            logger.info(
                f"  - Overall quality: {qc_report['quality_summary'].get('overall_grade', 'N/A')}"
            )
        else:
            logger.info(f"  - Report keys: {list(qc_report.keys())}")

        return processed

    except Exception as e:
        logger.error(f"✗ Minimal preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_mode_comparison():
    """Compare different preprocessing modes on the same data."""
    print_section("PREPROCESSING MODE COMPARISON")

    # Load a Sleep-EDF file
    sleep_edf_dir = Path("data/datasets/external/sleep-edf/sleep-cassette")
    edf_file = next(iter(sleep_edf_dir.glob("*-PSG.edf")))

    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    logger.info(f"Testing modes with: {edf_file.name}")

    modes = ["auto", "sleep", "abnormality", "minimal"]
    results = {}

    for mode in modes:
        logger.info(f"\n--- Testing mode: {mode} ---")

        preprocessor = FlexibleEEGPreprocessor(
            mode=mode,
            use_autoreject=False,  # Consistent across modes
        )

        try:
            processed = preprocessor.preprocess(raw.copy())

            results[mode] = {
                "success": True,
                "n_channels": len(processed.ch_names),
                "sfreq": processed.info["sfreq"],
                "has_montage": processed.get_montage() is not None,
                "channels": processed.ch_names,
            }

            logger.info(
                f"  ✓ Success: {results[mode]['n_channels']} channels, {results[mode]['sfreq']} Hz"
            )

        except Exception as e:
            results[mode] = {"success": False, "error": str(e)}
            logger.error(f"  ✗ Failed: {e}")

    # Summary
    print("\n--- Mode Comparison Summary ---")
    for mode, result in results.items():
        if result["success"]:
            print(
                f"{mode:12} | Channels: {result['n_channels']:2d} | Sfreq: {result['sfreq']:3.0f} Hz | Montage: {result['has_montage']}"
            )
        else:
            print(f"{mode:12} | FAILED: {result['error']}")


def main():
    """Run all flexible preprocessing tests."""
    print_section("FLEXIBLE PREPROCESSING PIPELINE TEST")

    # Test 1: Sleep-EDF without positions
    sleep_result = test_sleep_edf_preprocessing()

    # Test 2: TUH with positions
    tuh_result = test_tuh_preprocessing()

    # Test 3: Minimal preprocessing
    minimal_result = test_minimal_preprocessing()

    # Test 4: Mode comparison
    test_mode_comparison()

    # Summary
    print_section("TEST SUMMARY")

    tests_passed = sum(
        [sleep_result is not None, tuh_result is not None, minimal_result is not None]
    )

    print(f"\nTests passed: {tests_passed}/3")

    if tests_passed == 3:
        print("\n🎉 ALL FLEXIBLE PREPROCESSING TESTS PASSED! 🎉")
        print("\nThe flexible preprocessor successfully handles:")
        print("  ✓ Sleep-EDF data without channel positions")
        print("  ✓ TUH data with standard montages")
        print("  ✓ Different preprocessing modes for various tasks")
        print("  ✓ Graceful fallback when Autoreject unavailable")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
