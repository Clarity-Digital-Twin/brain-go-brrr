"""REAL tests for linear probe - No bullshit mocking."""

import numpy as np
import pytest
import torch
import torch.nn as nn


class TestLinearProbe:
    """Test linear probe head for EEGPT."""

    def test_linear_probe_initialization(self):
        """Test linear probe initialization."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead as LinearProbe

        probe = LinearProbe(input_dim=768, num_classes=2, dropout=0.1)

        # Check layers exist
        assert hasattr(probe, "dropout")
        assert hasattr(probe, "classifier")
        assert isinstance(probe.dropout, nn.Dropout)
        assert isinstance(probe.classifier, nn.Linear)

        # Check dimensions
        assert probe.classifier.in_features == 768
        assert probe.classifier.out_features == 2

    def test_linear_probe_forward(self):
        """Test forward pass through probe."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead as LinearProbe

        probe = LinearProbe(input_dim=512, num_classes=2)

        # Create batch of features
        batch_size = 8
        features = torch.randn(batch_size, 512)

        # Forward pass
        logits = probe(features)

        # Check output shape
        assert logits.shape == (batch_size, 2)

        # Check gradient flow
        loss = logits.sum()
        loss.backward()
        assert probe.classifier.weight.grad is not None

    def test_linear_probe_dropout(self):
        """Test dropout is applied in training mode."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead as LinearProbe

        probe = LinearProbe(input_dim=256, num_classes=2, dropout=0.5)

        # Set to training mode
        probe.train()

        # Same input, different outputs due to dropout
        x = torch.ones(10, 256)
        out1 = probe(x)
        out2 = probe(x)

        # Should be different due to dropout
        assert not torch.allclose(out1, out2)

        # Set to eval mode
        probe.eval()

        # Same input, same output (no dropout)
        out3 = probe(x)
        out4 = probe(x)

        # Should be identical
        assert torch.allclose(out3, out4)

    def test_linear_probe_multiclass(self):
        """Test probe with multiple classes."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead as LinearProbe

        # 5-class classification
        probe = LinearProbe(input_dim=768, num_classes=5)

        features = torch.randn(16, 768)
        logits = probe(features)

        assert logits.shape == (16, 5)

        # Check softmax gives valid probabilities
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-6)

    def test_linear_probe_binary_decision(self):
        """Test binary classification decision boundary."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead as LinearProbe

        probe = LinearProbe(input_dim=128, num_classes=2)

        # Create separable data
        class0 = torch.randn(50, 128) - 1.0  # Shifted negative
        class1 = torch.randn(50, 128) + 1.0  # Shifted positive

        # Get predictions
        probe.eval()
        logits0 = probe(class0)
        logits1 = probe(class1)

        # Should produce different distributions
        mean0 = logits0.mean(dim=0)
        mean1 = logits1.mean(dim=0)

        # Means should be different
        assert not torch.allclose(mean0, mean1, atol=0.1)


class TestTwoLayerProbe:
    """Test two-layer probe variant."""

    @pytest.fixture(autouse=True)
    def skip_if_not_implemented(self):
        """Skip tests if module doesn't exist, xfail if it does but is broken."""
        pytest.importorskip(
            "brain_go_brrr.models.eegpt_two_layer_probe", reason="TwoLayerProbe module not found"
        )

    @pytest.mark.xfail(
        strict=True, reason="TwoLayerProbe not yet implemented correctly - see issue #XXX"
    )
    def test_two_layer_initialization(self):
        """Test two-layer probe initialization."""
        from brain_go_brrr.models.eegpt_two_layer_probe import (
            TwoLayerProbeHead as TwoLayerProbe,
        )

        probe = TwoLayerProbe(input_dim=768, hidden_dim=256, num_classes=2, dropout=0.1)

        # Check layers
        assert hasattr(probe, "fc1")
        assert hasattr(probe, "fc2")
        assert probe.fc1.out_features == 256
        assert probe.fc2.out_features == 2

    @pytest.mark.xfail(
        strict=True, reason="TwoLayerProbe not yet implemented correctly - see issue #XXX"
    )
    def test_two_layer_forward(self):
        """Test two-layer forward pass."""
        from brain_go_brrr.models.eegpt_two_layer_probe import (
            TwoLayerProbeHead as TwoLayerProbe,
        )

        probe = TwoLayerProbe(input_dim=768, hidden_dim=128, num_classes=2)

        x = torch.randn(32, 768)
        out = probe(x)

        assert out.shape == (32, 2)

        # Check intermediate activations exist
        probe.eval()
        with torch.no_grad():
            # Hook to capture intermediate
            activations = []

            def hook(module, input, output):
                activations.append(output)

            handle = probe.fc1.register_forward_hook(hook)
            _ = probe(x)
            handle.remove()

            assert len(activations) == 1
            assert activations[0].shape == (32, 128)


class TestProbeTraining:
    """Test probe training dynamics."""

    @pytest.mark.slow
    def test_probe_learning(self):
        """Test that probe can learn simple pattern."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead as LinearProbe

        # Seed for reproducibility and speed
        torch.manual_seed(0)
        np.random.seed(0)

        probe = LinearProbe(input_dim=10, num_classes=2)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.1)  # Higher LR for faster convergence
        criterion = nn.CrossEntropyLoss()

        # Create VERY simple linearly separable data (tiny dataset for speed)
        x_class0 = torch.randn(20, 10) - 2.0  # Even more separation
        x_class1 = torch.randn(20, 10) + 2.0

        x = torch.cat([x_class0, x_class1])
        y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])

        # Shuffle
        perm = torch.randperm(40)
        x = x[perm]
        y = y[perm]

        # Train for very few steps
        probe.train()
        initial_loss = None

        for _ in range(5):  # Reduced to 5 epochs
            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss

        # Check accuracy
        probe.eval()
        with torch.no_grad():
            logits = probe(x)
            preds = logits.argmax(dim=-1)
            acc = (preds == y).float().mean().item()

        # Should achieve decent accuracy on linearly separable data
        assert acc > 0.7  # Conservative threshold

    def test_probe_gradient_flow(self):
        """Test gradient flows through probe."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead as LinearProbe

        probe = LinearProbe(input_dim=256, num_classes=3)

        x = torch.randn(16, 256, requires_grad=True)
        y = torch.randint(0, 3, (16,))

        logits = probe(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        # Check gradients exist
        assert probe.classifier.weight.grad is not None
        assert probe.classifier.bias.grad is not None
        assert x.grad is not None

        # Gradients should be non-zero
        assert probe.classifier.weight.grad.abs().sum() > 0
        assert x.grad.abs().sum() > 0
