"""Test probe compatibility and consolidation for P2 technical debt."""

import torch
import torch.nn as nn

from brain_go_brrr.infra.ml_models.linear_probe import TwoLayerProbe
from brain_go_brrr.infra.ml_models.probe_factory import ProbeFactory as UnifiedProbe


class TestProbeCompatibility:
    """Test suite for probe consolidation (P2 technical debt)."""

    def test_unified_probe_matches_two_layer(self):
        """Unified probe should produce same results as TwoLayerProbe."""
        input_dim = 2048
        hidden_dim = 256
        output_dim = 2

        # Create both probes
        probe1 = TwoLayerProbe(input_dim, hidden_dim, output_dim)
        probe2 = UnifiedProbe.create(input_dim, hidden_dim, output_dim, architecture="two_layer")

        # They should be the same object type for now
        assert isinstance(probe2, TwoLayerProbe)

        # Test forward pass with same input
        torch.manual_seed(42)
        x = torch.randn(32, input_dim)

        # Set to eval mode for deterministic behavior
        probe1.eval()
        probe2.eval()

        # Since probe2 IS a TwoLayerProbe, copy weights to ensure same output
        probe2.load_state_dict(probe1.state_dict())

        out1 = probe1(x)
        out2 = probe2(x)

        torch.testing.assert_close(out1, out2)

    def test_unified_probe_supports_all_modes(self):
        """Unified probe should support linear and two-layer modes."""
        input_dim = 2048
        hidden_dim = 256
        output_dim = 2

        # Create probes in different modes
        probe_two_layer = UnifiedProbe.create(
            input_dim, hidden_dim, output_dim, architecture="two_layer"
        )
        probe_linear = UnifiedProbe.create(input_dim, hidden_dim, output_dim, architecture="linear")

        # Check that two-layer is TwoLayerProbe
        assert isinstance(probe_two_layer, TwoLayerProbe)

        # Check that linear is LinearProbeHead
        from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead

        assert isinstance(probe_linear, LinearProbeHead)

        # Both should work with same input shape
        x = torch.randn(32, input_dim)
        out_two = probe_two_layer(x)
        out_lin = probe_linear(x)

        assert out_two.shape == (32, output_dim)
        assert out_lin.shape == (32, output_dim)

    def test_state_dict_compatibility(self):
        """Ensure state_dict compatibility for checkpoint loading."""
        input_dim = 2048
        hidden_dim = 256
        output_dim = 2

        # Create original probe
        original = TwoLayerProbe(input_dim, hidden_dim, output_dim)

        # Create unified probe
        unified = UnifiedProbe.create(input_dim, hidden_dim, output_dim, architecture="two_layer")

        # State dicts should be compatible
        original_state = original.state_dict()
        unified.load_state_dict(original_state)

        # Outputs should match after loading state
        x = torch.randn(16, input_dim)
        original.eval()
        unified.eval()

        out_orig = original(x)
        out_unif = unified(x)

        torch.testing.assert_close(out_orig, out_unif)

    def test_backward_compatibility_imports(self):
        """Ensure backward compatibility with existing imports."""
        # Direct import should still work
        from brain_go_brrr.infra.ml_models.linear_probe import TwoLayerProbe as DirectImport

        # Factory should return same type
        factory_probe = UnifiedProbe.create(2048, 256, 2, architecture="two_layer")

        assert isinstance(factory_probe, DirectImport)

    def test_dropout_parameter_forwarding(self):
        """Test that dropout parameter is correctly forwarded."""
        probe_with_dropout = UnifiedProbe.create(
            input_dim=2048,
            hidden_dim=256,
            output_dim=2,
            architecture="two_layer",
            dropout=0.5,
        )

        # Check dropout is set (TwoLayerProbe stores it in net Sequential)
        has_dropout = False
        for module in probe_with_dropout.net.modules():
            if isinstance(module, nn.Dropout):
                has_dropout = True
                assert module.p == 0.5

        assert has_dropout, "Dropout layer not found in probe"
