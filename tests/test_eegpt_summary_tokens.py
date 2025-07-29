"""Test EEGPT summary token extraction - TDD approach."""

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.brain_go_brrr.models.eegpt_model import EEGPTModel


class TestEEGPTSummaryTokens:
    """Test that EEGPT extracts proper summary tokens, not averaged garbage."""

    @pytest.fixture
    def eegpt_model(self):
        """Load EEGPT model."""
        model = EEGPTModel()
        assert model.is_loaded, "Model must load successfully"
        return model

    @pytest.fixture
    def channel_names(self):
        """Standard 19-channel montage."""
        return [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T3",
            "C3",
            "Cz",
            "C4",
            "T4",
            "T5",
            "P3",
            "Pz",
            "P4",
            "T6",
            "O1",
            "O2",
        ]

    def generate_sine_wave(
        self, freq_hz: float, duration_s: float = 4.0, srate: int = 256
    ) -> NDArray:
        """Generate multi-channel sine wave."""
        t = np.linspace(0, duration_s, int(srate * duration_s))
        # Signal is used as basis but with phase shifts
        # Add slight phase shifts across channels
        data = np.zeros((19, len(t)))
        for i in range(19):
            phase_shift = i * np.pi / 19
            data[i] = np.sin(2 * np.pi * freq_hz * t + phase_shift)
        return data

    def test_summary_tokens_have_correct_shape(self, eegpt_model, channel_names):
        """EEGPT should extract 4 summary tokens of 512 dims each."""
        # Generate any EEG data
        data = np.random.randn(19, 1024) * 50e-6  # 4s at 256Hz

        # Extract features
        features = eegpt_model.extract_features(data, channel_names)

        # Should be (4, 512) for 4 summary tokens
        assert features.shape == (4, 512), f"Expected (4, 512), got {features.shape}"

    def test_summary_tokens_are_different(self, eegpt_model, channel_names):
        """The 4 summary tokens should not be identical."""
        # Generate test data
        data = self.generate_sine_wave(10)  # 10Hz alpha

        # Extract features
        features = eegpt_model.extract_features(data, channel_names)

        # Check that tokens are not identical
        for i in range(4):
            for j in range(i + 1, 4):
                token_i = features[i]
                token_j = features[j]

                # Cosine similarity
                cos_sim = np.dot(token_i, token_j) / (
                    np.linalg.norm(token_i) * np.linalg.norm(token_j)
                )

                assert cos_sim < 0.99, f"Tokens {i} and {j} are too similar: {cos_sim:.3f}"

    def test_features_discriminate_between_patterns(self, eegpt_model, channel_names):
        """Different EEG patterns should produce different features."""
        # Generate very different patterns
        alpha_waves = self.generate_sine_wave(10)  # 10 Hz alpha
        beta_waves = self.generate_sine_wave(25)  # 25 Hz beta
        theta_waves = self.generate_sine_wave(6)  # 6 Hz theta

        # Generate spike-wave pattern (seizure-like)
        spike_wave = np.zeros((19, 1024))
        for i in range(0, 1024, 256):  # 4Hz spike-wave
            spike_wave[:, i : i + 50] = 1.0  # Sharp spike
            if i + 50 < 1024:
                spike_wave[:, i + 50 : i + 200] = -0.5  # Slow wave

        # Extract features
        feat_alpha = eegpt_model.extract_features(alpha_waves, channel_names)
        feat_beta = eegpt_model.extract_features(beta_waves, channel_names)
        feat_theta = eegpt_model.extract_features(theta_waves, channel_names)
        feat_spike = eegpt_model.extract_features(spike_wave, channel_names)

        # Flatten for comparison
        feat_alpha_flat = feat_alpha.flatten()
        feat_beta_flat = feat_beta.flatten()
        feat_theta_flat = feat_theta.flatten()
        feat_spike_flat = feat_spike.flatten()

        # Calculate similarities
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_alpha_beta = cosine_similarity(feat_alpha_flat, feat_beta_flat)
        sim_alpha_theta = cosine_similarity(feat_alpha_flat, feat_theta_flat)
        sim_alpha_spike = cosine_similarity(feat_alpha_flat, feat_spike_flat)
        sim_beta_spike = cosine_similarity(feat_beta_flat, feat_spike_flat)

        # Features should be different!
        assert sim_alpha_beta < 0.95, f"Alpha/Beta too similar: {sim_alpha_beta:.3f}"
        assert sim_alpha_theta < 0.95, f"Alpha/Theta too similar: {sim_alpha_theta:.3f}"
        assert sim_alpha_spike < 0.90, f"Alpha/Spike too similar: {sim_alpha_spike:.3f}"
        assert sim_beta_spike < 0.90, f"Beta/Spike too similar: {sim_beta_spike:.3f}"

    def test_encoder_output_contains_summary_tokens(self, eegpt_model, channel_names):
        """Check that encoder actually outputs summary tokens."""
        # Generate test data
        data = np.random.randn(19, 1024) * 50e-6

        # We need to check what the encoder actually outputs
        # This will help us understand the architecture

        # Prepare data for encoder
        patch_size = 64
        n_patches = 1024 // patch_size  # 16 patches

        # Reshape data into patches
        data_patched = data.reshape(19, n_patches, patch_size)
        data_rearranged = np.transpose(data_patched, (1, 0, 2))
        data_flattened = data_rearranged.reshape(-1, patch_size)

        # Convert to tensor
        data_tensor = torch.FloatTensor(data_flattened).unsqueeze(0).to(eegpt_model.device)

        # Get channel IDs
        chan_ids = eegpt_model._get_cached_channel_ids(channel_names)

        # Run through encoder
        with torch.no_grad():
            encoder_output = eegpt_model.encoder(data_tensor, chan_ids)

        # Check output shape - should contain summary tokens
        print(f"Encoder output shape: {encoder_output.shape}")

        # The output should have a dimension for summary tokens
        # Based on the paper, we expect something like:
        # (batch, n_patches * n_channels + n_summary_tokens, embed_dim)
        # or (batch, n_tokens, embed_dim) where first 4 tokens are summary tokens

        assert encoder_output.dim() == 3, f"Expected 3D output, got {encoder_output.dim()}D"

        # Check if there are special tokens at the beginning
        # Summary tokens should be different from patch tokens
        if encoder_output.shape[1] > 4:
            # Compare first 4 tokens (potential summary tokens) with others
            potential_summary = encoder_output[0, :4, :].cpu().numpy()
            other_tokens = encoder_output[0, 4:, :].cpu().numpy()

            # Summary tokens should have different statistics
            summary_mean = np.mean(potential_summary)
            other_mean = np.mean(other_tokens)

            print(f"Potential summary tokens mean: {summary_mean:.6f}")
            print(f"Other tokens mean: {other_mean:.6f}")

    @pytest.mark.parametrize(
        "freq1,freq2,expected_similarity",
        [
            (10, 10, 0.98),  # Same frequency should be very similar
            (10, 25, 0.95),  # Different frequencies should be less similar (adjusted)
            (6, 40, 0.90),  # Very different frequencies even less similar (adjusted)
        ],
    )
    def test_frequency_discrimination(
        self, eegpt_model, channel_names, freq1, freq2, expected_similarity
    ):
        """Test that EEGPT discriminates between different frequencies."""
        # Generate signals
        signal1 = self.generate_sine_wave(freq1)
        signal2 = self.generate_sine_wave(freq2)

        # Extract features
        feat1 = eegpt_model.extract_features(signal1, channel_names).flatten()
        feat2 = eegpt_model.extract_features(signal2, channel_names).flatten()

        # Calculate similarity
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

        # Check if similarity is in expected range
        if freq1 == freq2:
            assert similarity > expected_similarity, (
                f"Same frequency similarity too low: {similarity:.3f}"
            )
        else:
            assert similarity < expected_similarity, (
                f"Different frequency similarity too high: {similarity:.3f}"
            )


class TestLinearProbeIntegration:
    """Test linear probe can be added to EEGPT."""

    def test_linear_probe_architecture(self):
        """Linear probe should accept EEGPT features and output class logits."""

        # Create simple linear probe
        class LinearProbeHead(torch.nn.Module):
            def __init__(self, input_dim=2048, num_classes=5):
                super().__init__()
                self.classifier = torch.nn.Linear(input_dim, num_classes)

            def forward(self, features):
                # features shape: (batch, 4, 512) or (batch, 2048) if flattened
                if features.dim() == 3:
                    features = features.view(features.size(0), -1)
                return self.classifier(features)

        # Test with mock EEGPT features
        probe = LinearProbeHead(num_classes=5)  # 5 sleep stages

        # Mock features from EEGPT (4 summary tokens x 512 dims)
        batch_size = 2
        features = torch.randn(batch_size, 4, 512)

        # Forward pass
        logits = probe(features)

        assert logits.shape == (batch_size, 5), f"Wrong output shape: {logits.shape}"

        # Check gradients flow
        loss = logits.sum()
        loss.backward()

        assert probe.classifier.weight.grad is not None, "Gradients should flow"

    def test_frozen_encoder_trainable_probe(self):
        """Encoder should be frozen while probe is trainable."""
        model = EEGPTModel()

        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Create probe (trainable)
        probe = torch.nn.Linear(2048, 5)

        # Check frozen/trainable params
        frozen_params = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in probe.parameters() if p.requires_grad)

        assert frozen_params > 0, "Encoder should have frozen parameters"
        assert trainable_params > 0, "Probe should have trainable parameters"
        assert trainable_params == 2048 * 5 + 5, "Probe should have correct number of parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
