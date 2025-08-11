"""Sharp tests targeting red (low coverage) modules.

Based on auditor feedback, these tests specifically target:
1. WindowExtractor edge cases 
2. LinearProbe initialization
"""

import numpy as np
import torch


class TestWindowExtractorEdgeCases:
    """Sharp tests for window extractor edge cases."""

    def test_window_extractor_empty_data(self):
        """Test WindowExtractor handles empty data gracefully."""
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=2.0)

        # Empty array should return empty windows
        empty_data = np.array([]).reshape(0, 0)
        windows = extractor.extract(empty_data, sfreq=256)
        assert len(windows) == 0

    def test_window_extractor_single_sample(self):
        """Test WindowExtractor with data shorter than window."""
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=2.0)

        # Single sample - too short for any window
        short_data = np.random.randn(19, 100)  # Less than 4s at 256Hz
        windows = extractor.extract(short_data, sfreq=256)
        assert len(windows) == 0

    def test_window_extractor_exact_window(self):
        """Test WindowExtractor with exactly one window of data."""
        from brain_go_brrr.core.window_extractor import WindowExtractor

        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=0.0)

        # Exactly 4 seconds at 256Hz
        exact_data = np.random.randn(19, 1024)
        windows = extractor.extract(exact_data, sfreq=256)
        assert len(windows) == 1
        assert windows[0].shape == (19, 1024)

    def test_window_extractor_multiple_windows(self):
        """Test WindowExtractor with multiple windows."""
        from brain_go_brrr.core.window_extractor import WindowExtractor

        # 4s windows with 2s overlap
        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=2.0)

        # 10 seconds of data should give us 4 windows
        # Windows at: 0-4s, 2-6s, 4-8s, 6-10s
        data = np.random.randn(19, 2560)  # 10s at 256Hz
        windows = extractor.extract(data, sfreq=256)
        assert len(windows) == 4
        assert all(w.shape == (19, 1024) for w in windows)


class TestLinearProbeInit:
    """Sharp tests for LinearProbe initialization paths."""

    def test_linear_probe_default_init(self):
        """Test LinearProbeHead with default initialization."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=2048, num_classes=2)
        assert probe.input_dim == 2048
        assert probe.num_classes == 2
        assert hasattr(probe, 'classifier')

        # Test forward pass
        x = torch.randn(16, 2048)
        output = probe(x)
        assert output.shape == (16, 2)

    def test_linear_probe_custom_init(self):
        """Test LinearProbeHead with custom initialization."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=512, num_classes=5, dropout=0.3)
        assert probe.input_dim == 512
        assert probe.num_classes == 5

        # Check dropout is applied in training mode
        probe.train()
        x = torch.randn(8, 512)
        output1 = probe(x)
        output2 = probe(x)
        # Outputs should differ due to dropout
        assert not torch.allclose(output1, output2)

    def test_linear_probe_eval_mode(self):
        """Test LinearProbeHead behavior in eval mode."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=128, num_classes=2, dropout=0.5)
        probe.eval()

        x = torch.randn(4, 128)
        output1 = probe(x)
        output2 = probe(x)
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)

    def test_linear_probe_gradient_flow(self):
        """Test LinearProbeHead allows gradient flow."""
        from brain_go_brrr.models.linear_probe import LinearProbeHead

        probe = LinearProbeHead(input_dim=256, num_classes=2)
        x = torch.randn(4, 256, requires_grad=True)

        output = probe(x)
        loss = output.sum()
        loss.backward()

        # Check gradients flow
        assert x.grad is not None
        assert probe.classifier.weight.grad is not None
