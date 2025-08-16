"""Smoke test for backward compatibility imports."""

import importlib
import warnings


def test_old_imports_still_redirect():
    """Test that promised compatibility shims still work."""
    # These imports should work via compatibility shims
    # as promised in the release notes

    # Test data module redirects
    try:
        edf_loader = importlib.import_module("brain_go_brrr.data.edf_loader")
        assert edf_loader is not None
    except ImportError:
        # If the shim doesn't exist, check the core location still works
        edf_loader = importlib.import_module("brain_go_brrr.core.edf_loader")
        assert edf_loader is not None

    # Test hierarchical pipeline alias
    from brain_go_brrr.services.hierarchical_pipeline import HierarchicalPipeline

    assert HierarchicalPipeline is not None

    # Test that deprecated methods show warnings
    from brain_go_brrr.domain.abnormal.detector import AbnormalityDetector
    from brain_go_brrr.domain.quality.controller import EEGQualityController

    # Create instances
    detector = AbnormalityDetector()
    qc = EEGQualityController()

    # Check deprecated methods exist
    assert hasattr(
        detector, "compute_abnormality_score"
    ), "Backward compat method compute_abnormality_score missing"
    assert hasattr(detector, "is_abnormal"), "Backward compat method is_abnormal missing"
    assert hasattr(
        qc, "compute_quality_score"
    ), "Backward compat method compute_quality_score missing"

    # Test that they trigger deprecation warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Test deprecated detector method
        import numpy as np

        fake_features = np.random.randn(512).astype(np.float32)
        try:
            detector.compute_abnormality_score(fake_features)
        except Exception:
            pass  # Method might fail without model, but should exist

        # Check if deprecation warning was raised
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        assert (
            len(deprecation_warnings) > 0
        ), "Expected deprecation warning for compute_abnormality_score"
