"""Smoke test for factory function contracts."""

import importlib
import warnings


def test_factories_present_and_callable():
    """Test that documented factory functions exist and are callable."""
    # Filter DeprecationWarnings to ensure we catch any issues
    warnings.filterwarnings("error", category=DeprecationWarning)

    # Import the factories module
    factories = importlib.import_module("brain_go_brrr.application.factories")

    # Check core factories exist
    assert hasattr(factories, "create_quality_controller"), "Missing create_quality_controller"
    assert callable(factories.create_quality_controller)

    assert hasattr(factories, "create_abnormality_detector"), "Missing create_abnormality_detector"
    assert callable(factories.create_abnormality_detector)

    assert hasattr(factories, "create_sleep_analyzer"), "Missing create_sleep_analyzer"
    assert callable(factories.create_sleep_analyzer)

    assert hasattr(factories, "create_feature_extractor"), "Missing create_feature_extractor"
    assert callable(factories.create_feature_extractor)

    # Test that they can be called with defaults (no required args)
    # Sleep analyzer should work without args
    sleep_analyzer = factories.create_sleep_analyzer()
    assert sleep_analyzer is not None

    # Quality controller should work with None model_path (uses default)
    qc = factories.create_quality_controller()
    assert qc is not None

    # Abnormality detector should work with None model_path (uses default)
    detector = factories.create_abnormality_detector()
    assert detector is not None
