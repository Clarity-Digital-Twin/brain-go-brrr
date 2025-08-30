"""Tests for NaN handling policy - REAL BEHAVIORAL TESTS, NO MOCKING."""

import numpy as np
import pytest
import torch

from brain_go_brrr.domain.preprocessing.nan_policy import sanitize_data, validate_no_nan


class TestValidateNoNaN:
    """Test validate_no_nan function behavior."""

    def test_validate_clean_numpy_array(self):
        """Test validation passes for clean numpy array."""
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = validate_no_nan(data, "test_data")
        # Should return same array
        assert result is data
        assert np.array_equal(result, data)

    def test_validate_clean_torch_tensor(self):
        """Test validation passes for clean torch tensor."""
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        result = validate_no_nan(data, "test_tensor")
        # Should return same tensor
        assert result is data
        assert torch.equal(result, data)

    def test_validate_raises_on_nan_numpy(self):
        """Test validation raises on NaN in numpy array."""
        data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN detected in test_input"):
            validate_no_nan(data, "test_input")

    def test_validate_raises_on_inf_numpy(self):
        """Test validation raises on Inf in numpy array."""
        data = np.array([1.0, np.inf, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Inf detected in test_input"):
            validate_no_nan(data, "test_input")

    def test_validate_raises_on_nan_torch(self):
        """Test validation raises on NaN in torch tensor."""
        data = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="NaN detected in tensor_data"):
            validate_no_nan(data, "tensor_data")

    def test_validate_raises_on_inf_torch(self):
        """Test validation raises on Inf in torch tensor."""
        data = torch.tensor([1.0, float("inf"), 3.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="Inf detected in tensor_data"):
            validate_no_nan(data, "tensor_data")

    def test_validate_unsupported_type(self):
        """Test validation raises on unsupported data type."""
        data = [1.0, 2.0, 3.0]  # Plain list
        with pytest.raises(TypeError, match="Unsupported data type"):
            validate_no_nan(data)

    def test_validate_negative_inf(self):
        """Test validation catches negative infinity."""
        data_np = np.array([1.0, -np.inf, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Inf detected"):
            validate_no_nan(data_np)

        data_torch = torch.tensor([1.0, float("-inf"), 3.0])
        with pytest.raises(ValueError, match="Inf detected"):
            validate_no_nan(data_torch)


class TestSanitizeData:
    """Test sanitize_data function behavior."""

    def test_sanitize_zero_method_numpy(self):
        """Test zero replacement for NaN/Inf in numpy."""
        data = np.array([1.0, np.nan, 3.0, np.inf, 5.0], dtype=np.float32)
        result = sanitize_data(data, method="zero", name="test")

        # NaN and Inf should be replaced with 0
        expected = np.array([1.0, 0.0, 3.0, 0.0, 5.0], dtype=np.float32)
        assert np.allclose(result, expected)
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

    def test_sanitize_zero_method_torch(self):
        """Test zero replacement for NaN/Inf in torch."""
        data = torch.tensor([1.0, float("nan"), 3.0, float("inf"), 5.0])
        result = sanitize_data(data, method="zero", name="test")

        # NaN and Inf should be replaced with 0
        expected = torch.tensor([1.0, 0.0, 3.0, 0.0, 5.0])
        assert torch.allclose(result, expected)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_sanitize_median_method_1d(self):
        """Test median replacement for 1D array."""
        data = np.array([1.0, np.nan, 3.0, np.inf, 5.0], dtype=np.float32)
        result = sanitize_data(data, method="median", name="test")

        # Median of valid values [1, 3, 5] is 3.0
        expected = np.array([1.0, 3.0, 3.0, 3.0, 5.0], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_sanitize_median_method_2d(self):
        """Test channel-wise median replacement for 2D array."""
        data = np.array(
            [[1.0, np.nan, 3.0], [4.0, np.inf, 6.0], [7.0, 8.0, np.nan]], dtype=np.float32
        )
        result = sanitize_data(data, method="median", name="test")

        # Check no NaN/Inf remain
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

        # Channel 0: median of [1, 3] = 2.0
        assert result[0, 1] == 2.0
        # Channel 1: median of [4, 6] = 5.0
        assert result[1, 1] == 5.0
        # Channel 2: median of [7, 8] = 7.5
        assert result[2, 2] == 7.5

    def test_sanitize_mean_method_1d(self):
        """Test mean replacement for 1D array."""
        data = np.array([1.0, np.nan, 3.0, np.inf, 5.0], dtype=np.float32)
        result = sanitize_data(data, method="mean", name="test")

        # Mean of valid values [1, 3, 5] is 3.0
        expected = np.array([1.0, 3.0, 3.0, 3.0, 5.0], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_sanitize_mean_method_2d(self):
        """Test channel-wise mean replacement for 2D array."""
        data = np.array(
            [[1.0, np.nan, 3.0], [4.0, np.inf, 6.0], [7.0, 8.0, np.nan]], dtype=np.float32
        )
        result = sanitize_data(data, method="mean", name="test")

        # Check no NaN/Inf remain
        assert not np.isnan(result).any()
        assert not np.isinf(result).any()

        # Channel 0: mean of [1, 3] = 2.0
        assert result[0, 1] == 2.0
        # Channel 1: mean of [4, 6] = 5.0
        assert result[1, 1] == 5.0
        # Channel 2: mean of [7, 8] = 7.5
        assert result[2, 2] == 7.5

    def test_sanitize_invalid_method(self):
        """Test invalid sanitization method raises error."""
        data = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown sanitization method: invalid"):
            sanitize_data(data, method="invalid")

    def test_sanitize_unsupported_type(self):
        """Test sanitization with unsupported type raises error."""
        data = [1.0, 2.0, 3.0]  # Plain list
        with pytest.raises(TypeError, match="Unsupported data type"):
            sanitize_data(data)

    def test_sanitize_all_nan_channel(self):
        """Test sanitization handles channel with all NaN values."""
        data = np.array(
            [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]], dtype=np.float32
        )

        # With zero method, all NaNs should become 0
        result_zero = sanitize_data(data.copy(), method="zero", name="test")
        assert not np.isnan(result_zero).any()
        assert np.allclose(result_zero[1], [0.0, 0.0, 0.0])

        # With median/mean method, if all values in a channel are NaN,
        # the result will still be NaN (can't compute median/mean of nothing)
        # This is expected behavior - document it
        result_median = sanitize_data(data.copy(), method="median", name="test")
        # Middle channel will still be NaN
        assert np.isnan(result_median[1]).all()

    def test_sanitize_preserves_device(self):
        """Test torch tensor sanitization preserves device."""
        if torch.cuda.is_available():
            data = torch.tensor([1.0, float("nan"), 3.0]).cuda()
            result = sanitize_data(data, method="zero")
            assert result.device == data.device
            assert not torch.isnan(result).any()

    def test_sanitize_preserves_dtype(self):
        """Test sanitization preserves data type."""
        # Numpy float32
        data_f32 = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        result_f32 = sanitize_data(data_f32, method="zero")
        assert result_f32.dtype == np.float32

        # Numpy float64
        data_f64 = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        result_f64 = sanitize_data(data_f64, method="zero")
        assert result_f64.dtype == np.float64

        # Torch float32
        data_torch = torch.tensor([1.0, float("nan"), 3.0], dtype=torch.float32)
        result_torch = sanitize_data(data_torch, method="zero")
        assert result_torch.dtype == torch.float32


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_array(self):
        """Test handling of empty arrays."""
        data_np = np.array([], dtype=np.float32)
        result = validate_no_nan(data_np)
        assert len(result) == 0

        data_torch = torch.tensor([], dtype=torch.float32)
        result = validate_no_nan(data_torch)
        assert len(result) == 0

    def test_single_value_arrays(self):
        """Test single-value arrays."""
        # Clean value
        data = np.array([5.0], dtype=np.float32)
        assert validate_no_nan(data)[0] == 5.0

        # NaN value
        data_nan = np.array([np.nan], dtype=np.float32)
        with pytest.raises(ValueError, match="NaN detected"):
            validate_no_nan(data_nan)

        # Sanitize single NaN
        result = sanitize_data(data_nan, method="zero")
        assert result[0] == 0.0

    def test_mixed_positive_negative_inf(self):
        """Test handling of both positive and negative infinity."""
        data = np.array([1.0, np.inf, -np.inf, 2.0], dtype=np.float32)
        result = sanitize_data(data, method="mean")

        # Mean of [1, 2] = 1.5
        expected = np.array([1.0, 1.5, 1.5, 2.0], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_large_arrays(self):
        """Test performance with large arrays."""
        # Create large array with some NaNs
        size = 10000
        data = np.random.randn(100, size).astype(np.float32)
        # Inject some NaNs
        data[::10, ::100] = np.nan

        # Should complete without issues
        result = sanitize_data(data, method="median")
        assert not np.isnan(result).any()
        assert result.shape == data.shape
