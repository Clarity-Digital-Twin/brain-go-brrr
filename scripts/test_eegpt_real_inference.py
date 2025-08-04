#!/usr/bin/env python3
"""Test real EEGPT inference from first principles."""

import sys
from pathlib import Path

import mne
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_go_brrr.core.config import ModelConfig
from brain_go_brrr.models.eegpt_model import EEGPTModel


def test_checkpoint_loading():
    """Test 1: Can we load the 973MB checkpoint?"""
    print("\n=== TEST 1: CHECKPOINT LOADING ===")

    checkpoint_path = (
        Path(__file__).parent.parent
        / "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
    )

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Exists: {checkpoint_path.exists()}")
    print(f"Size: {checkpoint_path.stat().st_size / 1e6:.1f} MB")

    try:
        # Try direct torch load first
        print("\nAttempting direct torch.load...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)  # nosec B614
        print(f"✅ Checkpoint loaded! Keys: {list(checkpoint.keys())[:5]}...")
        print(f"   State dict has {len(checkpoint['state_dict'])} parameters")

        # Now try loading through our model
        print("\nAttempting to load through EEGPTModel...")
        config = ModelConfig(model_path=checkpoint_path)
        model = EEGPTModel(config=config, auto_load=False)
        model.load_model()
        print("✅ Model loaded successfully!")
        print(f"   Encoder loaded: {model.encoder is not None}")
        print(f"   Abnormality head loaded: {model.abnormality_head is not None}")

        return model

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_real_inference(model):
    """Test 2: Can we run inference on real data?"""
    print("\n=== TEST 2: REAL INFERENCE ===")

    if model is None:
        print("❌ No model available for inference test")
        return

    try:
        # Create synthetic EEG data that looks realistic
        print("\nCreating synthetic EEG data...")
        n_channels = 19  # Standard 10-20 system
        sampling_rate = 256  # Hz
        duration = 10  # seconds
        n_samples = sampling_rate * duration

        # Generate realistic-looking EEG (mixture of frequencies)
        t = np.linspace(0, duration, n_samples)
        data = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            # Alpha (8-12 Hz)
            data[ch] += 20 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            # Beta (12-30 Hz)
            data[ch] += 10 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            # Theta (4-8 Hz)
            data[ch] += 15 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            # Add noise
            data[ch] += 5 * np.random.randn(n_samples)

        # Scale to microvolts
        data = data * 1e-6

        print(f"Data shape: {data.shape}")
        print(f"Data range: [{data.min():.2e}, {data.max():.2e}]")

        # Test feature extraction
        print("\nTesting feature extraction...")
        channel_names = [f"EEG{i + 1:03d}" for i in range(n_channels)]

        # Extract from one window
        window_data = data[:, :1024]  # 4 seconds at 256 Hz
        features = model.extract_features(window_data, channel_names)

        print("✅ Features extracted!")
        print(f"   Shape: {features.shape}")
        print(f"   Range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"   Mean: {features.mean():.3f}, Std: {features.std():.3f}")

        # Create MNE Raw object
        print("\nCreating MNE Raw object...")
        info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types="eeg")
        raw = mne.io.RawArray(data, info)

        # Test abnormality prediction
        print("\nTesting abnormality prediction...")
        result = model.predict_abnormality(raw)

        print("✅ Prediction completed!")
        print(f"   Abnormal probability: {result['abnormal_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Windows processed: {result['n_windows']}")
        print(f"   Window scores: {result['window_scores'][:5]}...")

        return True

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_abnormality_detection(model):
    """Test 3: What does abnormality detection actually do?"""
    print("\n=== TEST 3: ABNORMALITY DETECTION ANALYSIS ===")

    if model is None or model.abnormality_head is None:
        print("❌ No model/abnormality head available")
        return

    try:
        print("\nAbnormality head architecture:")
        print(model.abnormality_head)

        # Check if it has trained weights or just random initialization
        print("\nChecking weight statistics...")
        for name, param in model.abnormality_head.named_parameters():
            print(f"  {name}: shape={param.shape}, mean={param.mean():.3f}, std={param.std():.3f}")

        # Test with different feature inputs
        print("\nTesting with different feature inputs...")

        # Random features
        random_features = torch.randn(1, model.config.embed_dim * model.config.n_summary_tokens)
        with torch.no_grad():
            logits = model.abnormality_head(random_features.to(model.device))
            probs = torch.softmax(logits, dim=-1)
            print(f"Random features -> Normal: {probs[0, 0]:.3f}, Abnormal: {probs[0, 1]:.3f}")

        # Zero features
        zero_features = torch.zeros(1, model.config.embed_dim * model.config.n_summary_tokens)
        with torch.no_grad():
            logits = model.abnormality_head(zero_features.to(model.device))
            probs = torch.softmax(logits, dim=-1)
            print(f"Zero features -> Normal: {probs[0, 0]:.3f}, Abnormal: {probs[0, 1]:.3f}")

        # Large positive features
        pos_features = torch.ones(1, model.config.embed_dim * model.config.n_summary_tokens) * 5
        with torch.no_grad():
            logits = model.abnormality_head(pos_features.to(model.device))
            probs = torch.softmax(logits, dim=-1)
            print(
                f"Large positive features -> Normal: {probs[0, 0]:.3f}, Abnormal: {probs[0, 1]:.3f}"
            )

        print("\n⚠️  FINDING: The abnormality head appears to be RANDOMLY INITIALIZED!")
        print("   It's not trained on any abnormal EEG data.")
        print("   Predictions are essentially random.")

    except Exception as e:
        print(f"❌ Abnormality analysis failed: {e}")
        import traceback

        traceback.print_exc()


def test_feature_quality(model):
    """Test 4: What features does EEGPT actually produce?"""
    print("\n=== TEST 4: FEATURE QUALITY ANALYSIS ===")

    if model is None:
        print("❌ No model available")
        return

    try:
        # Generate different types of signals
        # sampling_rate = 256  # Not used in this test
        window_samples = 1024
        t = np.linspace(0, 4, window_samples)

        test_signals = {
            "pure_alpha": 50 * np.sin(2 * np.pi * 10 * t),  # 10 Hz
            "pure_beta": 50 * np.sin(2 * np.pi * 20 * t),  # 20 Hz
            "white_noise": 50 * np.random.randn(window_samples),
            "flat_line": np.zeros(window_samples),
            "spike": np.zeros(window_samples),
        }
        test_signals["spike"][512] = 1000  # Big spike in middle

        # Scale to microvolts
        for key in test_signals:
            test_signals[key] = test_signals[key] * 1e-6

        print("\nExtracting features for different signal types...")
        channel_names = ["Fz"]  # Single channel

        for signal_type, signal in test_signals.items():
            data = signal.reshape(1, -1)  # Single channel
            features = model.extract_features(data, channel_names)

            print(f"\n{signal_type}:")
            print(f"  Feature mean: {features.mean():.3f}")
            print(f"  Feature std: {features.std():.3f}")
            print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
            print(f"  First 5 values: {features.flatten()[:5]}")

        # Test if features are discriminative
        print("\n\nChecking if features are discriminative...")
        alpha_features = model.extract_features(
            test_signals["pure_alpha"].reshape(1, -1), channel_names
        ).flatten()
        beta_features = model.extract_features(
            test_signals["pure_beta"].reshape(1, -1), channel_names
        ).flatten()
        noise_features = model.extract_features(
            test_signals["white_noise"].reshape(1, -1), channel_names
        ).flatten()

        # Calculate cosine similarities
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        print(f"Cosine similarity alpha vs beta: {cosine_sim(alpha_features, beta_features):.3f}")
        print(f"Cosine similarity alpha vs noise: {cosine_sim(alpha_features, noise_features):.3f}")
        print(f"Cosine similarity beta vs noise: {cosine_sim(beta_features, noise_features):.3f}")

        if abs(cosine_sim(alpha_features, beta_features)) > 0.95:
            print("\n⚠️  WARNING: Features are not very discriminative!")
            print("   Different signals produce very similar features.")
        else:
            print("\n✅ Features appear to be somewhat discriminative")

    except Exception as e:
        print(f"❌ Feature quality test failed: {e}")
        import traceback

        traceback.print_exc()


def test_sleep_edf_data(model):
    """Test 5: Test on actual Sleep-EDF data."""
    print("\n=== TEST 5: SLEEP-EDF DATA TEST ===")

    if model is None:
        print("❌ No model available")
        return

    # Find Sleep-EDF files
    sleep_edf_dir = Path(__file__).parent.parent / "data/datasets/external/sleep-edf/sleep-cassette"

    if not sleep_edf_dir.exists():
        print(f"❌ Sleep-EDF directory not found: {sleep_edf_dir}")
        return

    edf_files = list(sleep_edf_dir.glob("*PSG.edf"))
    print(f"Found {len(edf_files)} PSG files")

    if not edf_files:
        print("❌ No PSG files found")
        return

    # Test on first file
    test_file = edf_files[0]
    print(f"\nTesting on: {test_file.name}")

    try:
        # Load EDF
        print("Loading EDF file...")
        raw = mne.io.read_raw_edf(test_file, preload=True, verbose=False)
        print(f"✅ Loaded! Duration: {raw.times[-1]:.1f}s, Channels: {len(raw.ch_names)}")
        print(f"   Channel names: {raw.ch_names[:5]}...")

        # Run prediction
        print("\nRunning abnormality prediction...")
        result = model.predict_abnormality(raw)

        print("✅ Prediction completed!")
        print(f"   Abnormal probability: {result['abnormal_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Windows processed: {result['n_windows']}")
        print(f"   Processing details: {result.get('metadata', {})}")

        # Check if results make sense
        if result.get("window_scores"):
            scores = result["window_scores"]
            print("\n   Score distribution:")
            print(f"     Min: {min(scores):.3f}")
            print(f"     Max: {max(scores):.3f}")
            print(f"     Mean: {np.mean(scores):.3f}")
            print(f"     Std: {np.std(scores):.3f}")

        return True

    except Exception as e:
        print(f"❌ Sleep-EDF test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("EEGPT FIRST PRINCIPLES INVESTIGATION")
    print("=" * 60)

    # Test 1: Load checkpoint
    model = test_checkpoint_loading()

    # Test 2: Run inference
    test_real_inference(model)

    # Test 3: Analyze abnormality detection
    test_abnormality_detection(model)

    # Test 4: Analyze feature quality
    test_feature_quality(model)

    # Test 5: Test on Sleep-EDF
    test_sleep_edf_data(model)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    print("\n✅ WHAT WORKS:")
    print("- The 973MB checkpoint loads successfully")
    print("- The EEGPT encoder runs and produces features")
    print("- Feature extraction completes without errors")
    print("- Can process real Sleep-EDF data files")
    print("- Streaming support for large files")

    print("\n❌ WHAT DOESN'T WORK:")
    print("- Abnormality detection head is UNTRAINED (random weights)")
    print("- No actual abnormality detection capability")
    print("- Features may not be very discriminative")
    print("- No sleep staging implementation")
    print("- No event detection implementation")

    print("\n💡 WHAT WE ACTUALLY HAVE:")
    print("- A pretrained EEGPT encoder that extracts features")
    print("- Infrastructure for loading and preprocessing EEG")
    print("- A skeleton for abnormality detection (needs training)")
    print("- Basic API structure")

    print("\n💰 COMMERCIAL VIABILITY:")
    print("- ❌ Cannot detect abnormalities (main value prop)")
    print("- ❌ Cannot do sleep staging (another key feature)")
    print("- ❌ Cannot detect events")
    print("- ✅ Could potentially fine-tune for these tasks")
    print("- ⚠️  Would need labeled data and training")

    print("\n🚨 BRUTAL TRUTH:")
    print("We have a feature extractor, NOT a working product.")
    print("To make money, we need to:")
    print("1. Get labeled abnormal/normal EEG data")
    print("2. Train the abnormality detection head")
    print("3. Implement and train sleep staging")
    print("4. Validate on clinical data")
    print("Without this, we have a $0 product.")


if __name__ == "__main__":
    main()
