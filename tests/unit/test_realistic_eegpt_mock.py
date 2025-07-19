"""SKIPPED: Complex EEGPT model mocking (Anti-pattern).

This test file attempts to create "realistic" EEGPT embeddings through mocking,
which violates ML testing best practices:

1. Mock embeddings don't reflect real EEGPT learned representations
2. Tests fail due to unexpected keyword arguments (normality_strength)
3. Embedding value assertions are arbitrary and brittle
4. High maintenance - breaks when model interface changes
5. False confidence - tests pass but don't validate real model behavior

BETTER APPROACH:
- Test EEGPT integration with small real EEG samples
- Test preprocessing and postprocessing logic separately
- Use dependency injection for testable architecture
- Mock only I/O operations, not ML model internals

See docs/TESTING_BEST_PRACTICES.md for detailed guidance.

Reference: Eugene Yan - "Don't Mock Machine Learning Models In Unit Tests"
https://eugeneyan.com/writing/unit-testing-ml/
"""

import pytest

# Skip entire module with clear explanation
pytestmark = pytest.mark.skip(
    reason="Complex ML model mocking anti-pattern - use real data instead"
)

# Original problematic tests commented out for reference:
"""
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from tests.fixtures.mock_eegpt import (
    create_deterministic_embeddings,
    MockEEGPTModel,
)


class TestRealisticEEGPTMock:
    # These tests fail because:
    # 1. Mock doesn't match real EEGPT interface (unexpected kwargs)
    # 2. Embedding assertions are arbitrary
    # 3. Deterministic behavior doesn't reflect real model complexity

    def test_mock_eegpt_dimensions(self):
        # PROBLEM: Testing mock dimensions doesn't validate real model
        pass

    def test_deterministic_behavior(self):
        # PROBLEM: Real EEGPT has stochastic elements that mocks miss
        pass

    def test_embedding_extraction_mocked(self):
        # PROBLEM: Keyword argument 'normality_strength' not expected
        pass
"""

# RECOMMENDED REPLACEMENT using real integration tests:
"""
# In tests/integration/test_eegpt_integration.py:

@pytest.mark.integration
def test_eegpt_embedding_extraction():
    '''Test EEGPT with actual small EEG sample'''
    import mne
    from pathlib import Path

    # Use real 5-second EEG sample
    eeg_path = Path("tests/fixtures/eeg/tuab_001_norm_5s.fif")
    raw = mne.io.read_raw_fif(eeg_path, preload=True)

    # Test real EEGPT model (or skip if not available)
    try:
        eegpt_model = EEGPTModel.load_pretrained()
        embeddings = eegpt_model.extract_features(raw)

        # Test actual behavior, not mock behavior
        assert embeddings.shape[1] == 512  # Real embedding dimension
        assert not torch.isnan(embeddings).any()
        assert torch.isfinite(embeddings).all()

    except (FileNotFoundError, ImportError):
        pytest.skip("EEGPT model not available - use real model for integration tests")


@pytest.mark.unit
def test_preprocessing_without_model():
    '''Test preprocessing logic independently'''
    raw_data = create_synthetic_eeg(n_channels=19, duration=5)

    preprocessor = EEGPreprocessor()
    processed = preprocessor.prepare_for_eegpt(raw_data)

    # Test preprocessing logic, not model behavior
    assert processed.shape[0] == 19  # Channel count preserved
    assert processed.shape[1] == 1280  # 5 seconds * 256 Hz
    assert np.abs(processed.mean()) < 0.1  # Roughly centered
"""
