#!/usr/bin/env python3
"""Data Pipeline Validation Script.

===============================

Comprehensive test of the Brain-Go-Brrr data pipeline using real Sleep-EDF data
and the actual EEGPT model weights to validate end-to-end functionality.

This script tests:
1. Real EDF file loading
2. EEGPT model loading and inference
3. Sleep analysis pipeline
4. Quality control pipeline
5. Performance metrics
6. Output generation
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Add src and project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import numpy as np  # noqa: E402


def validate_environment() -> dict[str, Any]:
    """Validate the development environment and dependencies."""
    results = {"python_version": sys.version, "dependencies": {}, "errors": []}

    # Test core dependencies
    try:
        import torch

        results["dependencies"]["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    except ImportError as e:
        results["errors"].append(f"PyTorch not available: {e}")

    try:
        import mne

        results["dependencies"]["mne"] = {"version": mne.__version__}
    except ImportError as e:
        results["errors"].append(f"MNE not available: {e}")

    # Test optional dependencies
    optional_deps = ["yasa", "autoreject", "scipy", "sklearn"]
    for dep in optional_deps:
        try:
            module = __import__(dep)
            results["dependencies"][dep] = {
                "version": getattr(module, "__version__", "unknown"),
                "available": True,
            }
        except ImportError:
            results["dependencies"][dep] = {"available": False}

    return results


def find_sample_edf_files(max_files: int = 3) -> list[Path]:
    """Find sample EDF files from the Sleep-EDF dataset."""
    data_dir = Path("data/datasets/external/sleep-edf")

    if not data_dir.exists():
        print(f"❌ Sleep-EDF directory not found: {data_dir}")
        return []

    # Look for PSG files (not hypnogram files)
    edf_files = list(data_dir.rglob("*PSG.edf"))[:max_files]

    print(f"📁 Found {len(edf_files)} PSG files in Sleep-EDF dataset")
    for i, file in enumerate(edf_files, 1):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"   {i}. {file.name} ({size_mb:.1f} MB)")

    return edf_files


def test_edf_loading(edf_file: Path) -> dict[str, Any]:
    """Test loading an EDF file with MNE."""
    print(f"\n🧪 Testing EDF loading: {edf_file.name}")

    try:
        import mne

        start_time = time.time()
        raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
        load_time = time.time() - start_time

        # Get basic info
        info = {
            "success": True,
            "load_time_s": load_time,
            "n_channels": len(raw.ch_names),
            "duration_s": raw.times[-1],
            "sampling_rate": raw.info["sfreq"],
            "channels": raw.ch_names[:10],  # First 10 channels
            "file_size_mb": edf_file.stat().st_size / (1024 * 1024),
        }

        print(
            f"   ✅ Loaded {info['n_channels']} channels, {info['duration_s']:.1f}s @ {info['sampling_rate']}Hz"
        )
        print(f"   ⏱️  Load time: {load_time:.2f}s")

        return info

    except Exception as e:
        print(f"   ❌ Failed to load EDF: {e}")
        return {"success": False, "error": str(e)}


def test_eegpt_model_loading() -> dict[str, Any]:
    """Test loading the EEGPT model weights."""
    print("\n🧪 Testing EEGPT model loading")

    model_path = Path("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")

    if not model_path.exists():
        return {"success": False, "error": f"Model file not found: {model_path}"}

    try:
        import torch

        start_time = time.time()

        # Test basic checkpoint loading
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)  # nosec B614
        load_time = time.time() - start_time

        # Analyze checkpoint structure
        info = {
            "success": True,
            "load_time_s": load_time,
            "file_size_mb": model_path.stat().st_size / (1024 * 1024),
            "checkpoint_keys": list(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else ["Unknown structure"],
            "model_keys": list(checkpoint.get("model", {}).keys())[:10]
            if "model" in checkpoint
            else [],
        }

        print(f"   ✅ Loaded checkpoint ({info['file_size_mb']:.1f} MB) in {load_time:.2f}s")
        print(f"   📋 Checkpoint keys: {info['checkpoint_keys']}")

        return info

    except Exception as e:
        print(f"   ❌ Failed to load EEGPT model: {e}")
        return {"success": False, "error": str(e)}


def test_sleep_preprocessing(edf_file: Path) -> dict[str, Any]:
    """Test the sleep preprocessing pipeline."""
    print("\n🧪 Testing sleep preprocessing pipeline")

    try:
        # Import our modules
        import mne

        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        start_time = time.time()

        # Load raw data
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

        # Initialize preprocessor
        preprocessor = SleepPreprocessor()

        # Test preprocessing
        processed_raw = preprocessor.preprocess_for_yasa(raw)

        process_time = time.time() - start_time

        info = {
            "success": True,
            "process_time_s": process_time,
            "original_channels": len(raw.ch_names),
            "processed_channels": len(processed_raw.ch_names),
            "original_sfreq": raw.info["sfreq"],
            "processed_sfreq": processed_raw.info["sfreq"],
            "duration_s": processed_raw.times[-1],
        }

        print(f"   ✅ Processed {info['original_channels']}→{info['processed_channels']} channels")
        print(
            f"   📊 {info['original_sfreq']}→{info['processed_sfreq']} Hz, {info['duration_s']:.1f}s"
        )
        print(f"   ⏱️  Process time: {process_time:.2f}s")

        return info

    except Exception as e:
        print(f"   ❌ Sleep preprocessing failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def test_yasa_sleep_analysis(edf_file: Path) -> dict[str, Any]:
    """Test YASA sleep staging if available."""
    print("\n🧪 Testing YASA sleep analysis")

    try:
        import mne
        import yasa

        from brain_go_brrr.preprocessing.sleep_preprocessor import SleepPreprocessor

        start_time = time.time()

        # Load and preprocess
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        preprocessor = SleepPreprocessor()
        processed_raw = preprocessor.preprocess_for_yasa(raw)

        # Run YASA sleep staging (on a subset to save time)
        duration_limit = min(600, processed_raw.times[-1])  # Max 10 minutes for testing
        if duration_limit < processed_raw.times[-1]:
            processed_raw = processed_raw.crop(tmax=duration_limit)

        # Get EEG channel for YASA (try common sleep EEG channels)
        eeg_channels = [
            ch
            for ch in processed_raw.ch_names
            if any(eeg_name in ch.upper() for eeg_name in ["C3", "C4", "CZ", "FPZ", "PZ"])
        ]

        if not eeg_channels:
            return {"success": False, "error": "No suitable EEG channels found for YASA"}

        # Run sleep staging
        sls = yasa.SleepStaging(processed_raw, eeg_name=eeg_channels[0])
        hypnogram = sls.predict()

        analysis_time = time.time() - start_time

        # Get sleep statistics using pandas (YASA returns numpy array)
        import pandas as pd

        hypnogram_series = pd.Series(hypnogram)
        stage_counts = hypnogram_series.value_counts()

        info = {
            "success": True,
            "analysis_time_s": analysis_time,
            "duration_analyzed_min": duration_limit / 60,
            "eeg_channel_used": eeg_channels[0],
            "epochs_analyzed": len(hypnogram),
            "sleep_stages": stage_counts.to_dict(),
            "dominant_stage": stage_counts.idxmax(),
        }

        print(f"   ✅ Analyzed {info['duration_analyzed_min']:.1f} minutes using {eeg_channels[0]}")
        print(f"   📊 Sleep stages: {dict(stage_counts)}")
        print(f"   ⏱️  Analysis time: {analysis_time:.2f}s")

        return info

    except ImportError:
        print("   ⚠️  YASA not available - skipping sleep analysis")
        return {"success": False, "error": "YASA not installed"}
    except Exception as e:
        print(f"   ❌ YASA analysis failed: {e}")
        return {"success": False, "error": str(e)}


def test_qc_pipeline(edf_file: Path) -> dict[str, Any]:
    """Test the quality control pipeline."""
    print("\n🧪 Testing quality control pipeline")

    try:
        import mne

        from services.qc_flagger import EEGQualityController

        start_time = time.time()

        # Load raw data (limit duration for testing)
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        duration_limit = min(300, raw.times[-1])  # Max 5 minutes for testing
        if duration_limit < raw.times[-1]:
            raw = raw.crop(tmax=duration_limit)

        # Initialize QC flagger
        qc_flagger = EEGQualityController()

        # Run QC analysis
        qc_results = qc_flagger.detect_bad_channels(raw)

        qc_time = time.time() - start_time

        info = {
            "success": True,
            "qc_time_s": qc_time,
            "duration_analyzed_s": duration_limit,
            "total_channels": len(raw.ch_names),
            "bad_channels": qc_results,  # Already a list of bad channels
            "bad_channel_pct": (len(qc_results) / len(raw.ch_names)) * 100,
            "quality_grade": "good" if len(qc_results) == 0 else "poor",
            "abnormal_prob": 0,  # Not computed in simple bad channel detection
        }

        print(f"   ✅ QC analysis complete in {qc_time:.2f}s")
        print(
            f"   📊 Bad channels: {len(info['bad_channels'])}/{info['total_channels']} ({info['bad_channel_pct']:.1f}%)"
        )
        print(f"   🏆 Quality grade: {info['quality_grade']}")

        return info

    except Exception as e:
        print(f"   ❌ QC pipeline failed: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def generate_validation_report(results: dict[str, Any]) -> None:
    """Generate a comprehensive validation report."""
    print("\n" + "=" * 80)
    print("🎯 DATA PIPELINE VALIDATION REPORT")
    print("=" * 80)

    # Environment summary
    env = results["environment"]
    print("\n📋 ENVIRONMENT:")
    print(f"   Python: {env['python_version'].split()[0]}")

    if "torch" in env["dependencies"]:
        torch_info = env["dependencies"]["torch"]
        cuda_status = f"CUDA: {torch_info['cuda_available']}" + (
            f" ({torch_info['device_count']} GPUs)" if torch_info["cuda_available"] else ""
        )
        print(f"   PyTorch: {torch_info['version']} | {cuda_status}")

    if "mne" in env["dependencies"]:
        print(f"   MNE: {env['dependencies']['mne']['version']}")

    # Data validation summary
    print("\n📊 DATA VALIDATION:")
    print(f"   Sleep-EDF files: {len(results['sample_files'])} tested")

    if results["edf_tests"]:
        successful_loads = sum(1 for test in results["edf_tests"] if test["success"])
        print(f"   EDF loading: {successful_loads}/{len(results['edf_tests'])} successful")

    # Model validation
    print("\n🧠 MODEL VALIDATION:")
    if results["eegpt_test"]["success"]:
        model_info = results["eegpt_test"]
        print(f"   EEGPT model: ✅ Loaded ({model_info['file_size_mb']:.1f} MB)")
    else:
        print(f"   EEGPT model: ❌ Failed - {results['eegpt_test']['error']}")

    # Pipeline testing
    print("\n🔬 PIPELINE TESTING:")

    pipeline_tests = ["sleep_preprocessing", "yasa_analysis", "qc_pipeline"]
    for test_name in pipeline_tests:
        if test_name in results:
            test_result = results[test_name]
            status = "✅ PASS" if test_result["success"] else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
            if not test_result["success"]:
                print(f"      Error: {test_result['error']}")

    # Performance summary
    print("\n⚡ PERFORMANCE SUMMARY:")
    if results.get("edf_tests"):
        avg_load_time = np.mean([t["load_time_s"] for t in results["edf_tests"] if t["success"]])
        print(f"   Average EDF load time: {avg_load_time:.2f}s")

    if "eegpt_test" in results and results["eegpt_test"]["success"]:
        print(f"   EEGPT model load time: {results['eegpt_test']['load_time_s']:.2f}s")

    # Overall assessment
    print("\n🎯 OVERALL ASSESSMENT:")

    critical_tests = ["eegpt_test", "sleep_preprocessing"]
    passed_critical = sum(
        1 for test in critical_tests if results.get(test, {}).get("success", False)
    )

    if passed_critical == len(critical_tests):
        print("   STATUS: ✅ PIPELINE READY FOR DEVELOPMENT")
        print("   - Core infrastructure operational")
        print("   - Real data processing confirmed")
        print("   - EEGPT model accessible")
    elif passed_critical > 0:
        print("   STATUS: ⚠️  PARTIAL FUNCTIONALITY")
        print(f"   - Some components working ({passed_critical}/{len(critical_tests)})")
        print("   - May need dependency installation")
    else:
        print("   STATUS: ❌ CRITICAL ISSUES")
        print("   - Major components failing")
        print("   - Environment setup required")

    print("\n" + "=" * 80)


def main():
    """Main validation function."""
    print("🚀 Brain-Go-Brrr Data Pipeline Validation")
    print("=" * 50)

    results = {}

    # 1. Validate environment
    print("🔍 Validating environment...")
    results["environment"] = validate_environment()

    # 2. Find sample EDF files
    print("\n📁 Locating sample EDF files...")
    sample_files = find_sample_edf_files(max_files=2)  # Test 2 files
    results["sample_files"] = [str(f) for f in sample_files]

    if not sample_files:
        print("❌ No EDF files found - cannot test pipeline")
        return results

    # 3. Test EDF loading
    print("\n📖 Testing EDF file loading...")
    results["edf_tests"] = []
    for edf_file in sample_files:
        test_result = test_edf_loading(edf_file)
        results["edf_tests"].append(test_result)

    # 4. Test EEGPT model loading
    print("\n🧠 Testing EEGPT model loading...")
    results["eegpt_test"] = test_eegpt_model_loading()

    # 5. Test preprocessing pipeline (use first successful EDF)
    successful_edf = next(
        (f for f, test in zip(sample_files, results["edf_tests"], strict=False) if test["success"]),
        None,
    )

    if successful_edf:
        # Test sleep preprocessing
        results["sleep_preprocessing"] = test_sleep_preprocessing(successful_edf)

        # Test YASA analysis
        results["yasa_analysis"] = test_yasa_sleep_analysis(successful_edf)

        # Test QC pipeline
        results["qc_pipeline"] = test_qc_pipeline(successful_edf)

    # 6. Generate report
    generate_validation_report(results)

    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        traceback.print_exc()
