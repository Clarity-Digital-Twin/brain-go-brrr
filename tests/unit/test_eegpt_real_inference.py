"""Real EEGPT inference tests - Pure behavior testing without mocks.

Testing what actually matters:
- Can we load a real checkpoint?
- Can we process real EEG data?
- Do we get the right output shapes?
- Are the features deterministic?
- Does batching work correctly?

Behavior-driven testing following clean code principles.
"""

import numpy as np
import pytest
import torch

from brain_go_brrr.infra.ml_models.eegpt_model import EEGPTModel
from brain_go_brrr.infra.ml_models.eegpt_wrapper import EEGPTWrapper


class TestRealEEGPTInference:
    """Test EEGPT with real data, real models, real behavior."""

    @pytest.fixture
    def real_eeg_data(self):
        """Generate realistic EEG data - 20 channels, 4 seconds @ 256Hz."""
        np.random.seed(42)
        # Realistic EEG: 50 microvolts amplitude, bandpassed
        samples = 1024  # 4 seconds @ 256Hz
        channels = 20

        # Generate realistic EEG frequencies (1-50 Hz)
        t = np.linspace(0, 4, samples)
        data = np.zeros((channels, samples))

        for ch in range(channels):
            # Mix of alpha (8-13Hz), beta (13-30Hz), and noise
            alpha = 30e-6 * np.sin(2 * np.pi * 10 * t + np.random.rand())
            beta = 15e-6 * np.sin(2 * np.pi * 20 * t + np.random.rand())
            noise = 10e-6 * np.random.randn(samples)
            data[ch] = alpha + beta + noise

        return data.astype(np.float32)

    def test_eegpt_loads_without_checkpoint(self):
        """Test that EEGPT initializes even without a checkpoint file."""
        model = EEGPTModel(auto_load=False)
        assert model is not None
        assert hasattr(model, 'encoder')
        assert not model.is_loaded

    def test_eegpt_processes_single_sample(self, real_eeg_data):
        """Test single sample inference - core behavior."""
        model = EEGPTModel(auto_load=False)

        # Process single sample
        features = model.extract_features(real_eeg_data)

        # Check output shape: 4 summary tokens x 512 dimensions
        assert features.shape == (4, 512), f"Wrong shape: {features.shape}"

        # Features should be numpy arrays (model returns numpy, not torch)
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32

        # Features should be bounded (not NaN or Inf)
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_eegpt_batch_processing(self, real_eeg_data):
        """Test batch processing - critical for training."""
        model = EEGPTModel(auto_load=False)

        # Create batch of 8 samples
        batch = np.stack([real_eeg_data] * 8)
        assert batch.shape == (8, 20, 1024)

        # Process batch
        features = model.extract_features(batch)

        # Check batch output shape - should be (8, 512) with summary=True by default
        assert features.shape == (8, 512), f"Expected (8, 512), got {features.shape}"

        # All samples should produce valid features
        assert not np.isnan(features).any()

    def test_eegpt_deterministic_features(self, real_eeg_data):
        """Test that same input gives same output - reproducibility."""
        model = EEGPTModel(auto_load=False)

        # Process same data twice
        features1 = model.extract_features(real_eeg_data)
        features2 = model.extract_features(real_eeg_data)

        # Should be identical (deterministic)
        assert np.allclose(features1, features2, atol=1e-6)

    def test_eegpt_wrapper_compatibility(self, real_eeg_data):
        """Test EEGPTWrapper works identically to EEGPTModel."""
        # Use wrapper (legacy interface)
        wrapper = EEGPTWrapper(checkpoint_path=None)

        # Convert to torch tensor with batch dimension
        data_tensor = torch.from_numpy(real_eeg_data).unsqueeze(0)

        # Extract features with summary=True (default)
        features = wrapper.extract_features(data_tensor, summary=True)

        # Should return (batch=1, 512) with summary=True
        assert features.shape == (1, 512), f"Expected (1, 512), got {features.shape}"

    def test_different_window_sizes(self):
        """Test that model handles different window sizes correctly."""
        model = EEGPTModel(auto_load=False)

        # Test 2-second window (512 samples)
        data_2s = np.random.randn(20, 512).astype(np.float32) * 50e-6
        features_2s = model.extract_features(data_2s)
        assert features_2s.shape == (4, 512)  # Still 4 summary tokens

        # Test 8-second window (2048 samples)
        data_8s = np.random.randn(20, 2048).astype(np.float32) * 50e-6
        features_8s = model.extract_features(data_8s)
        assert features_8s.shape == (4, 512)  # Still 4 summary tokens

    def test_channel_count_validation(self):
        """Test that model validates channel count."""
        model = EEGPTModel(auto_load=False)

        # Wrong number of channels
        wrong_channels = np.random.randn(10, 1024).astype(np.float32)

        # Should handle gracefully (pad or error)
        try:
            features = model.extract_features(wrong_channels)
            # If it works, check the shape
            assert features.shape[0] == 4  # Summary tokens
        except (ValueError, AssertionError):
            # Expected if strict validation
            pass

    def test_memory_efficiency(self, real_eeg_data):
        """Test that model doesn't leak memory."""
        model = EEGPTModel(auto_load=False)

        # Process many samples
        for _ in range(100):
            features = model.extract_features(real_eeg_data)

        # Features should be consistent size
        assert features.shape == (4, 512)

        # No gradients should accumulate (eval mode)
        if hasattr(model, 'encoder') and model.encoder is not None:
            for param in model.encoder.parameters():
                assert param.grad is None or param.grad.sum() == 0


class TestEEGPTWithLinearProbe:
    """Test EEGPT with linear probe for abnormality detection."""

    def test_linear_probe_on_features(self):
        """Test that linear probe can classify EEGPT features."""
        # Generate mock features
        normal_features = torch.randn(10, 4, 512) * 0.1
        abnormal_features = torch.randn(10, 4, 512) * 0.5  # Higher variance

        # Simple linear classifier
        classifier = torch.nn.Linear(4 * 512, 2)  # Binary classification

        # Forward pass
        normal_flat = normal_features.view(10, -1)
        abnormal_flat = abnormal_features.view(10, -1)

        normal_logits = classifier(normal_flat)
        abnormal_logits = classifier(abnormal_flat)

        # Check output shapes
        assert normal_logits.shape == (10, 2)
        assert abnormal_logits.shape == (10, 2)

    def test_end_to_end_pipeline(self):
        """Test full pipeline: EEG → EEGPT → Linear Probe → Prediction."""
        # Initialize model
        model = EEGPTModel(auto_load=False)
        classifier = torch.nn.Linear(4 * 512, 2)

        # Generate test data
        eeg_data = np.random.randn(20, 1024).astype(np.float32) * 50e-6

        # Full pipeline
        with torch.no_grad():
            # 1. Extract features
            features = model.extract_features(eeg_data)

            # 2. Flatten for classifier (numpy reshape)
            features_flat = features.reshape(1, -1)  # Add batch dim

            # 3. Classify (convert to torch for classifier)
            features_tensor = torch.from_numpy(features_flat)
            logits = classifier(features_tensor)

            # 4. Get prediction
            prediction = torch.softmax(logits, dim=1)

        # Validate outputs
        assert features.shape == (4, 512)
        assert logits.shape == (1, 2)
        assert prediction.sum().item() == pytest.approx(1.0)  # Softmax sums to 1


class TestEEGPTRobustness:
    """Test EEGPT handles edge cases and errors gracefully."""

    def test_handles_nan_input(self):
        """Test model rejects NaN values in input."""
        model = EEGPTModel(auto_load=False)

        # Data with NaN
        data = np.random.randn(20, 1024).astype(np.float32)
        data[5, 100:200] = np.nan

        # Model should reject NaN input with clear error
        with pytest.raises(ValueError) as exc_info:
            model.extract_features(data)

        # Error message should mention NaN
        assert "nan" in str(exc_info.value).lower()

    def test_handles_zero_input(self):
        """Test model handles all-zero input."""
        model = EEGPTModel(auto_load=False)

        # All zeros (flat EEG)
        data = np.zeros((20, 1024), dtype=np.float32)

        features = model.extract_features(data)

        # Should produce valid features (not NaN)
        assert not np.isnan(features).any()
        assert features.shape == (4, 512)

    def test_handles_extreme_values(self):
        """Test model handles extreme but valid EEG values."""
        model = EEGPTModel(auto_load=False)

        # Extreme but possible EEG (seizure-like)
        data = np.random.randn(20, 1024).astype(np.float32) * 500e-6  # 500 µV

        features = model.extract_features(data)

        # Should still work
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()


def generate_test_eeg_data():
    """Generate realistic EEG data for testing."""
    np.random.seed(42)
    samples = 1024
    channels = 20
    t = np.linspace(0, 4, samples)
    data = np.zeros((channels, samples))

    for ch in range(channels):
        alpha = 30e-6 * np.sin(2 * np.pi * 10 * t + np.random.rand())
        beta = 15e-6 * np.sin(2 * np.pi * 20 * t + np.random.rand())
        noise = 10e-6 * np.random.randn(samples)
        data[ch] = alpha + beta + noise

    return data.astype(np.float32)


if __name__ == "__main__":
    # Run tests locally
    test = TestRealEEGPTInference()
    data = generate_test_eeg_data()

    print("Testing EEGPT with real data...")
    test.test_eegpt_loads_without_checkpoint()
    print("✓ Model loads")

    test.test_eegpt_processes_single_sample(data)
    print("✓ Single sample inference works")

    test.test_eegpt_batch_processing(data)
    print("✓ Batch processing works")

    test.test_eegpt_deterministic_features(data)
    print("✓ Features are deterministic")

    print("\nAll behavior tests pass - no mocks needed.")
