"""Tests for LinearProbeHead initialization and functionality."""

import pytest
import torch

# Set seed for reproducibility
torch.manual_seed(42)


class TestLinearProbeHead:
    """Tests for LinearProbeHead model component."""

    def test_default_initialization(self):
        """Test LinearProbeHead with default initialization."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=2048, num_classes=2)
        assert probe.input_dim == 2048
        assert probe.num_classes == 2
        assert hasattr(probe, "classifier")

        # Test forward pass
        x = torch.randn(16, 2048)
        with torch.no_grad():
            output = probe(x)
        assert output.shape == (16, 2)

    def test_custom_initialization(self):
        """Test LinearProbeHead with custom parameters."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=512, num_classes=5, dropout=0.3)
        assert probe.input_dim == 512
        assert probe.num_classes == 5
        assert hasattr(probe, "dropout")

    def test_dropout_behavior_in_training(self):
        """Test that dropout is active in training mode."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=256, num_classes=3, dropout=0.5)
        probe.train()

        # Use fixed input but different random states for dropout
        x = torch.ones(8, 256)  # Fixed input

        torch.manual_seed(1)
        with torch.no_grad():
            output1 = probe(x)

        torch.manual_seed(2)
        with torch.no_grad():
            output2 = probe(x)

        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2, atol=1e-5)

    def test_no_dropout_in_eval_mode(self):
        """Test that dropout is disabled in eval mode."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=128, num_classes=2, dropout=0.5)
        probe.eval()

        x = torch.randn(4, 128)
        with torch.no_grad():
            output1 = probe(x)
            output2 = probe(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=256, num_classes=2)
        x = torch.randn(4, 256, requires_grad=True)

        output = probe(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert probe.classifier.weight.grad is not None
        assert probe.classifier.bias.grad is not None

    @pytest.mark.parametrize(
        "input_dim,num_classes", [(128, 2), (256, 5), (512, 10), (1024, 2), (2048, 3)]
    )
    def test_various_dimensions(self, input_dim, num_classes):
        """Test LinearProbeHead with various input/output dimensions."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=input_dim, num_classes=num_classes)

        batch_size = 8
        x = torch.randn(batch_size, input_dim)

        with torch.no_grad():
            output = probe(x)

        assert output.shape == (batch_size, num_classes)

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises appropriate error."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=256, num_classes=2)

        # Wrong input dimension should raise error
        wrong_input = torch.randn(4, 128)  # 128 != 256

        with pytest.raises(RuntimeError):
            probe(wrong_input)

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compatibility(self):
        """Test LinearProbeHead works on GPU."""
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=512, num_classes=2)
        probe = probe.cuda()

        x = torch.randn(8, 512).cuda()

        with torch.no_grad():
            output = probe(x)

        assert output.is_cuda
        assert output.shape == (8, 2)
