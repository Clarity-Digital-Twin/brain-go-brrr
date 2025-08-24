"""Strict tests for EEGPT compatibility layer shape contracts.

Tests the fail-fast behavior and explicit shape contracts.
"""

import warnings

import numpy as np
import pytest
import torch

from brain_go_brrr.infra.ml_models.eegpt_compat import EEGPTModel


class TestStrictShapeContracts:
    """Test strict shape contracts with compat_coerce=False."""

    def test_unexpected_shape_raises_without_coerce(self):
        """Test that unexpected shapes raise ValueError when compat_coerce=False."""
        # Create model with compat_coerce=False (default)
        model = EEGPTModel(auto_load=False, compat_coerce=False)
        model.is_loaded = True

        # Mock encoder that returns unexpected shape
        class BadEncoder:
            def extract_features(self, x, summary=True):
                # Return wrong shape (1, 768) instead of (1, 512)
                return torch.zeros(1, 768)

        model.encoder = BadEncoder()

        # Single sample data
        data = np.random.randn(19, 1024).astype(np.float32)

        # Should raise ValueError for unexpected shape
        with pytest.raises(ValueError, match="Unexpected summary shape"):
            model.extract_features(data, summary=True)

    def test_packed_tokens_with_coerce(self):
        """Test that packed tokens (1, 2048) are coerced properly with warning."""
        model = EEGPTModel(auto_load=False, compat_coerce=True)
        model.is_loaded = True

        # Mock encoder that returns packed tokens
        class PackedEncoder:
            def extract_features(self, x, summary=False):
                # Return packed tokens (1, 2048)
                return torch.zeros(1, 2048)

        model.encoder = PackedEncoder()

        data = np.random.randn(19, 1024).astype(np.float32)

        # Should coerce with deprecation warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features = model.extract_features(data, summary=False)

            # Check warning was raised
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Coercing packed tokens" in str(w[0].message)

        # Check output shape (4, 512) for legacy single sample
        assert features.shape == (4, 512)

    def test_summary_true_returns_batch_512(self):
        """Test that summary=True always returns (B, 512)."""
        model = EEGPTModel(auto_load=False, compat_coerce=False)
        model.is_loaded = True

        # Mock encoder that returns correct summary
        class GoodEncoder:
            def extract_features(self, x, summary=True):
                batch_size = x.shape[0]
                if summary:
                    return torch.zeros(batch_size, 512)
                else:
                    return torch.zeros(batch_size, 4, 512)

        model.encoder = GoodEncoder()

        # Test single sample
        data_single = np.random.randn(19, 1024).astype(np.float32)
        features = model.extract_features(data_single, summary=True)
        assert features.shape == (1, 512)

        # Test batch
        data_batch = np.random.randn(3, 19, 1024).astype(np.float32)
        features = model.extract_features(data_batch, summary=True)
        assert features.shape == (3, 512)

    def test_summary_false_returns_batch_4_512(self):
        """Test that summary=False returns (B, 4, 512)."""
        model = EEGPTModel(auto_load=False, compat_coerce=False)
        model.is_loaded = True

        # Mock encoder
        class GoodEncoder:
            def extract_features(self, x, summary=False):
                batch_size = x.shape[0]
                return torch.zeros(batch_size, 4, 512)

        model.encoder = GoodEncoder()

        # Test single sample - should keep batch dim without compat_coerce
        data_single = np.random.randn(19, 1024).astype(np.float32)
        features = model.extract_features(data_single, summary=False)
        assert features.shape == (1, 4, 512)

        # Test batch
        data_batch = np.random.randn(3, 19, 1024).astype(np.float32)
        features = model.extract_features(data_batch, summary=False)
        assert features.shape == (3, 4, 512)

    def test_tiling_summary_raises_without_coerce(self):
        """Test that tiling summary to tokens raises without compat_coerce."""
        model = EEGPTModel(auto_load=False, compat_coerce=False)
        model.is_loaded = True

        # Mock encoder that returns summary when tokens requested
        class WrongEncoder:
            def extract_features(self, x, summary=False):
                # Always return summary shape
                return torch.zeros(x.shape[0], 512)

        model.encoder = WrongEncoder()

        data = np.random.randn(19, 1024).astype(np.float32)

        # Should raise for wrong shape
        with pytest.raises(ValueError, match="Unexpected token shape"):
            model.extract_features(data, summary=False)

    def test_legacy_single_sample_with_coerce(self):
        """Test legacy single sample returns (4, 512) only with compat_coerce."""
        # With compat_coerce=True
        model_compat = EEGPTModel(auto_load=False, compat_coerce=True)
        model_compat.is_loaded = True

        class TokenEncoder:
            def extract_features(self, x, summary=False):
                return torch.zeros(x.shape[0], 4, 512)

        model_compat.encoder = TokenEncoder()

        data = np.random.randn(19, 1024).astype(np.float32)

        # Should return (4, 512) for legacy compatibility
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features = model_compat.extract_features(data, summary=False)
            assert features.shape == (4, 512)
            # Should have deprecation warning
            assert any("legacy single-sample" in str(warning.message) for warning in w)

        # Without compat_coerce - should keep batch
        model_strict = EEGPTModel(auto_load=False, compat_coerce=False)
        model_strict.is_loaded = True
        model_strict.encoder = TokenEncoder()

        features = model_strict.extract_features(data, summary=False)
        assert features.shape == (1, 4, 512)  # Keeps batch dimension


class TestPrepareForEEGPT:
    """Test prepare_for_eegpt assertions."""

    def test_padding_assertion(self):
        """Test that prepare_for_eegpt ensures T % 64 == 0."""
        import mne

        from brain_go_brrr.domain.preprocessing.eegpt_prepare import prepare_for_eegpt

        # Create raw with non-multiple of 64 samples
        sfreq = 256
        n_samples = 1000  # Not a multiple of 64
        data = np.random.randn(19, n_samples) * 1e-6
        ch_names = [f"CH{i}" for i in range(19)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # prepare_for_eegpt should pad it
        prepared_data = prepare_for_eegpt(raw, pad_to_multiple=64)

        # Check that output is padded to multiple of 64
        assert prepared_data.shape[1] % 64 == 0
        assert prepared_data.shape[1] == 1024  # Next multiple of 64

    def test_sampling_rate_assertion(self):
        """Test that prepare_for_eegpt ensures correct sampling rate."""
        import mne

        from brain_go_brrr.domain.preprocessing.eegpt_prepare import prepare_for_eegpt

        # Create raw with different sampling rate
        sfreq = 128  # Not 256
        n_samples = 512
        data = np.random.randn(19, n_samples) * 1e-6
        ch_names = [f"CH{i}" for i in range(19)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # prepare_for_eegpt should resample
        prepared_data = prepare_for_eegpt(raw, target_sfreq=256)

        # The function returns data array, but modifies raw in place
        # Check that raw was resampled (data shape will change)
        assert prepared_data.shape[1] == 1024  # 512 * 2 for 2x sampling rate

    def test_nan_validation(self):
        """Test that prepare_for_eegpt rejects NaN values."""
        import mne

        from brain_go_brrr.domain.preprocessing.eegpt_prepare import prepare_for_eegpt

        # Create raw with NaN
        sfreq = 256
        n_samples = 1024
        data = np.random.randn(19, n_samples) * 1e-6
        data[5, 100:200] = np.nan  # Add NaN values

        ch_names = [f"CH{i}" for i in range(19)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        # Should raise ValueError for NaN
        with pytest.raises(ValueError, match="NaN detected"):
            prepare_for_eegpt(raw)


class TestSupportsMethod:
    """Test supports() capability gates."""

    def test_sleep_analyzer_supports(self):
        """Test EnhancedSleepAnalyzer.supports() method."""
        import mne

        from brain_go_brrr.domain.sleep.analyzer_enhanced import EnhancedSleepAnalyzer

        analyzer = EnhancedSleepAnalyzer()

        # Valid EEG data
        sfreq = 256
        data = np.random.randn(19, sfreq * 60) * 1e-6  # 60 seconds
        ch_names = [f"CH{i}" for i in range(19)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)

        assert analyzer.supports(raw) is True

        # No EEG channels
        info_no_eeg = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='misc')
        raw_no_eeg = mne.io.RawArray(data, info_no_eeg)
        assert analyzer.supports(raw_no_eeg) is False

        # Too short duration
        data_short = np.random.randn(19, sfreq * 10) * 1e-6  # Only 10 seconds
        raw_short = mne.io.RawArray(data_short, info)
        assert analyzer.supports(raw_short) is False

        # Bad sampling rate
        info_bad_sfreq = mne.create_info(ch_names=ch_names, sfreq=10, ch_types='eeg')
        raw_bad_sfreq = mne.io.RawArray(data[:, : 10 * 60], info_bad_sfreq)
        assert analyzer.supports(raw_bad_sfreq) is False
