"""Property-based tests for EEGPT shape contracts using Hypothesis."""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from brain_go_brrr.domain.preprocessing.eegpt_preprocessing import prepare_for_eegpt
from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel


class TestEEGPTContractFuzzing:
    """Fuzz test shape contracts to catch silent drift early."""

    @given(
        n_samples=st.integers(min_value=64, max_value=4096).filter(lambda x: x % 64 == 0),
        batch_size=st.sampled_from([1, 2, 4, 8]),
        summary_mode=st.booleans(),
    )
    def test_extract_features_shape_contract(self, n_samples, batch_size, summary_mode):
        """Fuzz test that extract_features always returns exact expected shapes."""
        # Create model with mock encoder
        model = EEGPTModel(auto_load=False)
        model.is_loaded = True

        # Mock encoder that returns correct shapes
        class MockEncoder:
            def extract_features(self, x, summary=True):
                b = x.shape[0]
                if summary:
                    return torch.zeros(b, 512)
                else:
                    return torch.zeros(b, 4, 512)

        model.encoder = MockEncoder()

        # Generate test data
        if batch_size == 1:
            data = np.random.randn(20, n_samples).astype(np.float32)
            expected_batch = 1
        else:
            data = np.random.randn(batch_size, 20, n_samples).astype(np.float32)
            expected_batch = batch_size

        # Test extraction
        features = model.extract_features(data, summary=summary_mode)

        # Assert exact shape contract
        if summary_mode:
            assert features.shape == (
                expected_batch,
                512,
            ), f"Summary mode must return (B, 512), got {features.shape}"
        else:
            assert features.shape == (
                expected_batch,
                4,
                512,
            ), f"Token mode must return (B, 4, 512), got {features.shape}"

    @given(
        n_samples=st.integers(min_value=50, max_value=2000),
        target_multiple=st.sampled_from([64, 128, 256]),
    )
    @settings(deadline=None)  # MNE operations can be slow on first call
    @pytest.mark.slow
    def test_prepare_for_eegpt_padding_contract(self, n_samples, target_multiple):
        """Fuzz test that prepare_for_eegpt ensures T % target_multiple == 0."""
        import mne

        # Create raw EEG with arbitrary sample count
        sfreq = 256
        data = np.random.randn(19, n_samples) * 1e-6
        ch_names = [f"CH{i}" for i in range(19)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # Prepare with specific padding target
        prepared = prepare_for_eegpt(raw, pad_to_multiple=target_multiple)

        # Assert padding contract
        assert (
            prepared.shape[1] % target_multiple == 0
        ), f"prepare_for_eegpt must pad to multiple of {target_multiple}, got {prepared.shape[1]}"

        # Also verify it's the minimum valid padding
        assert prepared.shape[1] >= n_samples, "Padded size must be >= original size"
        assert (
            prepared.shape[1] < n_samples + target_multiple
        ), "Padded size must be minimal (< original + target_multiple)"

    @given(wrong_dim=st.sampled_from([768, 1024, 2048, 256]), summary_mode=st.booleans())
    def test_wrong_shapes_always_raise(self, wrong_dim, summary_mode):
        """Fuzz test that non-contract shapes always raise ValueError."""
        model = EEGPTModel(auto_load=False)
        model.is_loaded = True

        # Mock encoder that returns wrong dimensions
        class BadEncoder:
            def extract_features(self, x, summary=True):
                if summary:
                    return torch.zeros(1, wrong_dim)  # Wrong!
                else:
                    return torch.zeros(1, wrong_dim)  # Wrong!

        model.encoder = BadEncoder()
        data = np.random.randn(20, 1024).astype(np.float32)

        # Should always raise for non-contract shapes
        if wrong_dim != 512 or not summary_mode:
            with pytest.raises(ValueError, match="Unexpected"):
                model.extract_features(data, summary=summary_mode)
