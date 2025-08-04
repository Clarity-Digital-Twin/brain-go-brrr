"""Test linear probe implementation."""

import pytest
import torch

from brain_go_brrr.models.linear_probe import (
    AbnormalityProbe,
    LinearProbeHead,
    SleepStageProbe,
    create_probe_for_task,
)


class TestLinearProbeHead:
    """Test basic linear probe functionality."""

    def test_initialization(self):
        """Test probe initialization."""
        probe = LinearProbeHead(input_dim=2048, num_classes=5)

        assert probe.input_dim == 2048
        assert probe.num_classes == 5
        assert probe.classifier.weight.shape == (5, 2048)
        assert probe.classifier.bias.shape == (5,)

    def test_forward_pass_3d_input(self):
        """Test forward pass with 3D input (batch, n_tokens, embed_dim)."""
        probe = LinearProbeHead(input_dim=2048, num_classes=5)

        # Mock EEGPT features: batch_size=2, n_summary_tokens=4, embed_dim=512
        features = torch.randn(2, 4, 512)

        logits = probe(features)

        assert logits.shape == (2, 5)
        assert not torch.isnan(logits).any()

    def test_forward_pass_2d_input(self):
        """Test forward pass with 2D input (batch, flattened_features)."""
        probe = LinearProbeHead(input_dim=2048, num_classes=5)

        # Mock flattened features
        features = torch.randn(2, 2048)

        logits = probe(features)

        assert logits.shape == (2, 5)
        assert not torch.isnan(logits).any()

    def test_predict_proba(self):
        """Test probability prediction."""
        probe = LinearProbeHead(input_dim=2048, num_classes=5)

        features = torch.randn(3, 4, 512)
        probs = probe.predict_proba(features)

        assert probs.shape == (3, 5)
        assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-6)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_flow(self):
        """Test that gradients flow through the probe."""
        probe = LinearProbeHead(input_dim=2048, num_classes=5)

        features = torch.randn(2, 4, 512, requires_grad=True)
        labels = torch.tensor([0, 2])

        logits = probe(features)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        assert probe.classifier.weight.grad is not None
        assert probe.classifier.bias.grad is not None
        assert features.grad is not None


class TestSleepStageProbe:
    """Test sleep stage specific probe."""

    def test_initialization(self):
        """Test sleep probe has 5 classes."""
        probe = SleepStageProbe()

        assert probe.num_classes == 5
        assert len(probe.stage_names) == 5
        assert probe.stage_names == ["W", "N1", "N2", "N3", "REM"]

    def test_predict_stage(self):
        """Test stage prediction with names."""
        probe = SleepStageProbe()

        # Mock features for 3 samples
        features = torch.randn(3, 4, 512)

        stage_names, confidences = probe.predict_stage(features)

        assert len(stage_names) == 3
        assert all(stage in probe.stage_names for stage in stage_names)
        assert confidences.shape == (3,)
        assert (confidences >= 0).all() and (confidences <= 1).all()


class TestAbnormalityProbe:
    """Test abnormality detection probe."""

    def test_initialization(self):
        """Test abnormality probe has 2 classes."""
        probe = AbnormalityProbe()

        assert probe.num_classes == 2

    def test_predict_abnormal_probability(self):
        """Test abnormality probability prediction."""
        probe = AbnormalityProbe()

        features = torch.randn(5, 4, 512)
        abnormal_probs = probe.predict_abnormal_probability(features)

        assert abnormal_probs.shape == (5,)
        assert (abnormal_probs >= 0).all() and (abnormal_probs <= 1).all()


class TestProbeFactory:
    """Test probe factory function."""

    def test_create_sleep_probe(self):
        """Test creating sleep probe."""
        probe = create_probe_for_task("sleep")
        assert isinstance(probe, SleepStageProbe)
        assert probe.num_classes == 5

    def test_create_abnormality_probe(self):
        """Test creating abnormality probe."""
        probe = create_probe_for_task("abnormality")
        assert isinstance(probe, AbnormalityProbe)
        assert probe.num_classes == 2

    def test_create_generic_probe(self):
        """Test creating generic probe."""
        probe = create_probe_for_task("custom_task", num_classes=7)
        assert isinstance(probe, LinearProbeHead)
        assert probe.num_classes == 7

    def test_case_insensitive(self):
        """Test factory is case insensitive."""
        probe1 = create_probe_for_task("SLEEP")
        probe2 = create_probe_for_task("Sleep")
        probe3 = create_probe_for_task("sleep")

        assert all(isinstance(p, SleepStageProbe) for p in [probe1, probe2, probe3])


class TestIntegrationWithEEGPT:
    """Test integration with EEGPT features."""

    def test_end_to_end_sleep_classification(self):
        """Test full pipeline from EEGPT features to sleep stage."""
        # Mock EEGPT model output
        eegpt_features = torch.randn(1, 4, 512)  # 1 window, 4 summary tokens

        # Create and use probe
        probe = SleepStageProbe()
        stage_names, confidences = probe.predict_stage(eegpt_features)

        assert len(stage_names) == 1
        assert stage_names[0] in ["W", "N1", "N2", "N3", "REM"]
        assert 0 <= confidences[0] <= 1

    def test_batch_processing(self):
        """Test processing multiple windows."""
        # Mock batch of EEGPT features
        batch_size = 10
        eegpt_features = torch.randn(batch_size, 4, 512)

        probe = AbnormalityProbe()
        abnormal_probs = probe.predict_abnormal_probability(eegpt_features)

        assert abnormal_probs.shape == (batch_size,)
        assert (abnormal_probs >= 0).all() and (abnormal_probs <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
