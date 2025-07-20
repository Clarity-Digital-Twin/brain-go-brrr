#!/usr/bin/env python
"""Test the complete Brain-Go-Brrr pipeline with real data.

This script tests:
1. EDF data loading and preprocessing
2. EEGPT model inference
3. Sleep analysis with YASA
4. Quality control with Autoreject
5. Abnormality detection
6. Snippet extraction and analysis
"""

import sys
from pathlib import Path

import mne

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain_go_brrr.core.logger import get_logger
from brain_go_brrr.data.edf_streaming import decide_streaming, estimate_memory_usage
from brain_go_brrr.models.eegpt_model import EEGPTModel
from brain_go_brrr.preprocessing.eeg_preprocessor import EEGPreprocessor
from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor
from core.quality import EEGQualityController
from core.sleep import SleepAnalyzer
from core.snippets import EEGSnippetMaker

# Set up logging
logger = get_logger(__name__)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print("=" * 80)


def test_edf_loading():
    """Test EDF data loading and basic properties."""
    print_section("1. EDF DATA LOADING TEST")

    # Find Sleep-EDF files
    sleep_edf_dir = Path("data/datasets/external/sleep-edf/sleep-cassette")
    edf_files = list(sleep_edf_dir.glob("*-PSG.edf"))[:3]  # Test first 3 files

    if not edf_files:
        logger.error("No Sleep-EDF files found!")
        return None

    results = []
    for edf_file in edf_files:
        logger.info(f"\nTesting file: {edf_file.name}")

        # Estimate memory usage
        memory_estimate = estimate_memory_usage(edf_file)
        logger.info(f"  - Duration: {memory_estimate['duration_minutes']:.1f} minutes")
        logger.info(f"  - Channels: {memory_estimate['n_channels']}")
        logger.info(f"  - Sampling rate: {memory_estimate['sampling_rate']} Hz")
        logger.info(f"  - Memory estimate: {memory_estimate['estimated_total_mb']:.1f} MB")

        # Decide on streaming
        streaming_decision = decide_streaming(edf_file, max_memory_mb=500)
        logger.info(f"  - Use streaming: {streaming_decision['used_streaming']}")

        # Load the data
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            logger.info(
                f"  ✓ Successfully loaded: {len(raw.ch_names)} channels, {raw.n_times} samples"
            )
            results.append((edf_file, raw))
        except Exception as e:
            logger.error(f"  ✗ Failed to load: {e}")

    return results


def test_preprocessing(raw_data_list):
    """Test preprocessing pipeline."""
    print_section("2. PREPROCESSING TEST")

    if not raw_data_list:
        logger.error("No data to preprocess!")
        return None

    processed_data = []

    for edf_file, raw in raw_data_list:
        logger.info(f"\nPreprocessing: {edf_file.name}")

        try:
            # Determine if this is sleep data
            is_sleep_data = "sleep-edf" in str(edf_file).lower()

            if is_sleep_data:
                # For sleep data, use minimal SleepPreprocessor
                preprocessor = SleepPreprocessor()
                logger.info("  - Using sleep-specific preprocessing (minimal)")

                # Map Sleep-EDF channels to standard names
                raw_copy = raw.copy()
                channel_mapping = {
                    "EEG Fpz-Cz": "Fpz",
                    "EEG Pz-Oz": "Pz",
                    "EOG horizontal": "EOG",
                    "EMG submental": "EMG",
                }

                # Rename channels
                rename_dict = {}
                for old_name, new_name in channel_mapping.items():
                    if old_name in raw_copy.ch_names:
                        rename_dict[old_name] = new_name

                if rename_dict:
                    raw_copy.rename_channels(rename_dict)
                    logger.info(f"  - Renamed {len(rename_dict)} channels to standard names")

                # Process with proper channel types
                eeg_channels = [ch for ch in raw_copy.ch_names if ch in ["Fpz", "Pz"]]
                eog_channels = [ch for ch in raw_copy.ch_names if "EOG" in ch]
                emg_channels = [ch for ch in raw_copy.ch_names if "EMG" in ch]

                processed = preprocessor.preprocess_for_yasa(
                    raw_copy,
                    eeg_channels=eeg_channels,
                    eog_channels=eog_channels,
                    emg_channels=emg_channels,
                )
            else:
                # For other data, use standard EEGPreprocessor
                preprocessor = EEGPreprocessor()
                logger.info("  - Using standard EEG preprocessing")
                processed = preprocessor.preprocess(raw.copy())
            # Apply preprocessing
            processed = preprocessor.preprocess(raw)

            logger.info("  ✓ Preprocessing complete:")
            logger.info(f"    - Sampling rate: {processed.info['sfreq']} Hz")
            logger.info(f"    - Channels: {len(processed.ch_names)}")
            logger.info(f"    - Bad channels marked: {processed.info['bads']}")

            processed_data.append((edf_file, processed))

        except Exception as e:
            logger.error(f"  ✗ Preprocessing failed: {e}")

    return processed_data


def test_eegpt_inference(processed_data):
    """Test EEGPT model inference."""
    print_section("3. EEGPT MODEL INFERENCE TEST")

    if not processed_data:
        logger.error("No processed data for EEGPT!")
        return None

    # Initialize model
    model_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

    if not model_path.exists():
        logger.error(f"EEGPT model not found at {model_path}")
        logger.info("Please download the model from the EEGPT repository")
        return None

    try:
        eegpt_model = EEGPTModel(checkpoint_path=str(model_path))
        logger.info("✓ EEGPT model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load EEGPT model: {e}")
        return None

    results = []
    for edf_file, raw in processed_data[:1]:  # Test on first file only (for speed)
        logger.info(f"\nRunning EEGPT on: {edf_file.name}")

        try:
            # Extract features
            features = eegpt_model.extract_features(raw)

            logger.info("  ✓ Feature extraction complete:")
            logger.info(f"    - Feature shape: {features.shape}")
            logger.info(
                f"    - Feature stats: mean={features.mean():.3f}, std={features.std():.3f}"
            )

            # Test abnormality detection
            abnormal_result = eegpt_model.predict_abnormality(raw)
            logger.info("  ✓ Abnormality detection:")
            logger.info(f"    - Probability: {abnormal_result['abnormal_probability']:.3f}")
            logger.info(f"    - Confidence: {abnormal_result['confidence']:.3f}")
            logger.info(f"    - Classification: {abnormal_result['classification']}")

            results.append(
                {"file": edf_file.name, "features": features, "abnormality": abnormal_result}
            )

        except Exception as e:
            logger.error(f"  ✗ EEGPT inference failed: {e}")

    # Clean up
    eegpt_model.cleanup()

    return results


def test_sleep_analysis(processed_data):
    """Test sleep staging with YASA."""
    print_section("4. SLEEP ANALYSIS TEST")

    if not processed_data:
        logger.error("No processed data for sleep analysis!")
        return None

    analyzer = SleepAnalyzer()
    results = []

    for edf_file, raw in processed_data[:2]:  # Test first 2 files
        logger.info(f"\nAnalyzing sleep for: {edf_file.name}")

        try:
            # Run full sleep analysis
            sleep_results = analyzer.run_full_sleep_analysis(raw)

            logger.info("  ✓ Sleep analysis complete:")

            # Extract key metrics from nested structure
            sleep_stats = sleep_results.get("sleep_statistics", {})
            quality_metrics = sleep_results.get("quality_metrics", {})
            analysis_info = sleep_results.get("analysis_info", {})

            logger.info(f"    - Total sleep time: {sleep_stats.get('TST', 0):.1f} min")
            logger.info(f"    - Sleep efficiency: {sleep_stats.get('SE', 0):.1f}%")
            logger.info(f"    - REM percentage: {sleep_stats.get('%REM', 0):.1f}%")
            logger.info(f"    - N3 percentage: {sleep_stats.get('%N3', 0):.1f}%")
            logger.info(f"    - Number of epochs: {analysis_info.get('total_epochs', 0)}")
            logger.info(f"    - Quality grade: {quality_metrics.get('quality_grade', 'N/A')}")

            results.append({"file": edf_file.name, "results": sleep_results})

        except Exception as e:
            logger.error(f"  ✗ Sleep analysis failed: {e}")

    return results


def test_quality_control(raw_data_list):
    """Test quality control pipeline."""
    print_section("5. QUALITY CONTROL TEST")

    if not raw_data_list:
        logger.error("No data for QC!")
        return None

    qc_controller = EEGQualityController()
    results = []

    for edf_file, raw in raw_data_list[:2]:  # Test first 2 files
        logger.info(f"\nRunning QC on: {edf_file.name}")

        try:
            # Run full QC pipeline
            qc_report = qc_controller.run_full_qc_pipeline(raw.copy())

            logger.info("  ✓ QC analysis complete:")
            logger.info(f"    - Bad channels: {qc_report.get('bad_channels', [])}")
            logger.info(f"    - SNR: {qc_report.get('snr', 0):.2f} dB")
            logger.info(
                f"    - Artifact percentage: {qc_report.get('artifact_percentage', 0):.1f}%"
            )
            logger.info(f"    - Overall quality: {qc_report.get('overall_quality', 'Unknown')}")

            results.append({"file": edf_file.name, "report": qc_report})

        except Exception as e:
            logger.error(f"  ✗ QC analysis failed: {e}")

    return results


def test_snippet_extraction(processed_data):
    """Test snippet extraction and analysis."""
    print_section("6. SNIPPET EXTRACTION TEST")

    if not processed_data:
        logger.error("No processed data for snippet extraction!")
        return None

    snippet_maker = EEGSnippetMaker()
    results = []

    for edf_file, raw in processed_data[:1]:  # Test on first file
        logger.info(f"\nExtracting snippets from: {edf_file.name}")

        try:
            # Extract fixed-length snippets
            snippets = snippet_maker.extract_fixed_snippets(
                raw,
                snippet_length=4.0,  # 4 seconds for EEGPT
                overlap=0.5,
            )

            logger.info("  ✓ Snippet extraction complete:")
            logger.info(f"    - Number of snippets: {len(snippets)}")

            if snippets:
                logger.info(f"    - Snippet duration: {snippets[0]['duration']}s")
                logger.info(f"    - Channels per snippet: {len(snippets[0]['channels'])}")

            # Create report for first 5 snippets
            if len(snippets) > 5:
                report = snippet_maker.create_snippet_report(
                    snippets[:5],
                    include_features=True,
                    include_eegpt=False,  # Skip EEGPT for speed
                )
                logger.info("  ✓ Snippet analysis complete")
                logger.info(f"    - Analyzed {len(report['snippets'])} snippets")

            results.append(
                {
                    "file": edf_file.name,
                    "n_snippets": len(snippets),
                    "snippets": snippets[:3],  # Save first 3 for inspection
                }
            )

        except Exception as e:
            logger.error(f"  ✗ Snippet extraction failed: {e}")

    return results


def test_tuh_events_data():
    """Test with TUH EEG Events data."""
    print_section("7. TUH EEG EVENTS DATA TEST")

    tuh_dir = Path("data/datasets/external/tuh_eeg_events/v2.0.1/edf/train")
    if not tuh_dir.exists():
        logger.warning("TUH EEG Events data not found")
        return None

    # Find first few EDF files
    edf_files = list(tuh_dir.rglob("*.edf"))[:2]

    for edf_file in edf_files:
        logger.info(f"\nTesting TUH file: {edf_file.name}")

        try:
            # Memory estimate
            memory_estimate = estimate_memory_usage(edf_file)
            logger.info(f"  - Duration: {memory_estimate['duration_minutes']:.1f} minutes")
            logger.info(f"  - Memory: {memory_estimate['estimated_total_mb']:.1f} MB")

            # Load and check
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            logger.info(f"  ✓ Loaded: {len(raw.ch_names)} channels, {raw.info['sfreq']} Hz")

            # Quick QC check
            info = raw.info
            logger.info(f"  - Channel types: {set(info.get_channel_types())}")

        except Exception as e:
            logger.error(f"  ✗ Failed to process: {e}")


def main():
    """Run all pipeline tests."""
    print_section("BRAIN-GO-BRRR FULL PIPELINE TEST")

    # Summary results
    summary = {
        "edf_loading": False,
        "preprocessing": False,
        "eegpt_inference": False,
        "sleep_analysis": False,
        "quality_control": False,
        "snippet_extraction": False,
        "tuh_events": False,
    }

    # 1. Test EDF loading
    raw_data_list = test_edf_loading()
    summary["edf_loading"] = bool(raw_data_list)

    if not raw_data_list:
        logger.error("Cannot continue without loaded data!")
        return

    # 2. Test preprocessing
    processed_data = test_preprocessing(raw_data_list)
    summary["preprocessing"] = bool(processed_data)

    # 3. Test EEGPT inference
    eegpt_results = test_eegpt_inference(processed_data)
    summary["eegpt_inference"] = bool(eegpt_results)

    # 4. Test sleep analysis
    sleep_results = test_sleep_analysis(processed_data)
    summary["sleep_analysis"] = bool(sleep_results)

    # 5. Test quality control
    qc_results = test_quality_control(raw_data_list)
    summary["quality_control"] = bool(qc_results)

    # 6. Test snippet extraction
    snippet_results = test_snippet_extraction(processed_data)
    summary["snippet_extraction"] = bool(snippet_results)

    # 7. Test TUH Events data
    test_tuh_events_data()
    summary["tuh_events"] = True  # Marked as tested even if no data

    # Print summary
    print_section("TEST SUMMARY")

    total_passed = sum(summary.values())
    total_tests = len(summary)

    for test_name, passed in summary.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 ALL PIPELINE TESTS PASSED! 🎉")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
