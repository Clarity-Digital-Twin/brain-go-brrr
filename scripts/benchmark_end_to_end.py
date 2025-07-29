#!/usr/bin/env python3
"""Run end-to-end performance benchmark on Sleep-EDF data.

This script tests if we meet the performance target:
- Process a 30-minute EEG record in < 2 seconds
"""

import time
from pathlib import Path

import mne
import numpy as np

from brain_go_brrr.core.config import ModelConfig
from brain_go_brrr.models.eegpt_model import EEGPTModel


def main():
    """Run the benchmark."""
    print("🏃 Running end-to-end performance benchmark...")
    print("Target: Process 30-minute EEG record in < 2 seconds\n")

    # Use Sleep-EDF data
    data_path = Path("data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf")

    if not data_path.exists():
        print("❌ Sleep-EDF data not found. Creating synthetic 30-minute data...")
        # Create synthetic 30-minute recording
        sfreq = 256
        duration_minutes = 30
        n_channels = 19
        n_samples = int(sfreq * 60 * duration_minutes)

        # Create realistic EEG data
        data = np.random.randn(n_channels, n_samples) * 50  # ~50 µV amplitude
        ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
        info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info)
    else:
        print(f"✅ Loading Sleep-EDF data from: {data_path}")
        raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)

        # Crop to 30 minutes for consistent benchmark
        raw.crop(tmax=30 * 60)

    # Initialize model
    print("\n📊 Initializing EEGPT model...")
    config = ModelConfig(device="cpu")  # Use CPU for consistent benchmarking
    model = EEGPTModel(config=config)

    # Warm-up run (important for fair benchmarking)
    print("\n🔥 Warm-up run...")
    warm_up_data = raw.copy().crop(tmax=10)  # 10 second warm-up
    _ = model.predict_abnormality(warm_up_data)

    # Actual benchmark
    print("\n⏱️  Running benchmark...")
    start_time = time.perf_counter()

    result = model.predict_abnormality(raw)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Display results
    print("\n📊 BENCHMARK RESULTS:")
    print("=" * 50)
    print(f"Recording duration: {raw.times[-1] / 60:.1f} minutes")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Windows processed: {result.get('n_windows', 0)}")
    print(f"Used streaming: {result.get('used_streaming', False)}")
    print(f"\n⏱️  Processing time: {elapsed_time:.2f} seconds")
    print("🎯 Target: < 2.0 seconds")

    throughput = (raw.times[-1] / 60) / (elapsed_time / 60)

    if elapsed_time < 2.0:
        print(f"✅ PASS - Performance target met! ({elapsed_time:.2f}s < 2.0s)")
        print(f"📈 Throughput: {throughput:.1f}x real-time")
    else:
        print(f"❌ FAIL - Performance target not met ({elapsed_time:.2f}s > 2.0s)")
        print(f"   Need {elapsed_time / 2.0:.1f}x speedup")
        print(f"📈 Throughput: {throughput:.1f}x real-time")

    # Additional metrics
    if result.get("window_scores"):
        print("\n📊 Analysis results:")
        print(f"   Abnormal probability: {result['abnormal_probability']:.2%}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Mean window score: {result.get('mean_score', 0):.3f}")
        print(f"   Std window score: {result.get('std_score', 0):.3f}")

    print("\n" + "=" * 50)

    # Write results to file for CI integration
    results_file = Path("benchmark_results/end_to_end_performance.txt")
    results_file.parent.mkdir(exist_ok=True)

    with results_file.open("w") as f:
        f.write(f"duration_minutes: {raw.times[-1] / 60:.1f}\n")
        f.write(f"processing_time_seconds: {elapsed_time:.2f}\n")
        f.write("target_seconds: 2.0\n")
        f.write(f"passed: {'true' if elapsed_time < 2.0 else 'false'}\n")
        f.write(f"throughput_x_realtime: {throughput:.1f}\n")

    print(f"\n💾 Results saved to: {results_file}")

    return 0 if elapsed_time < 2.0 else 1


if __name__ == "__main__":
    exit(main())
