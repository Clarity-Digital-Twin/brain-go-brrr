"""Test environment variable configuration."""

import os
from pathlib import Path

import pytest


class TestEnvironmentConfig:
    """Test environment variable configuration."""

    @pytest.mark.skip(reason="EEGPT_MODEL_PATH no longer defined in api.main after refactor")
    def test_eegpt_model_path_from_env(self):
        """Test that EEGPT_MODEL_PATH can be set from environment."""
        # This test is outdated - model path is now passed as argument
        # to EEGPTModel instead of being a global constant
        pass

    def test_path_resolution(self):
        """Test that Path conversion works correctly."""
        test_path = "/custom/path/to/model.ckpt"

        # Test that Path conversion works correctly
        result = Path(test_path).absolute()
        assert result.is_absolute()
        assert str(result).endswith("model.ckpt")

    def test_environment_variables_available(self):
        """Test that we can set and read environment variables."""
        test_var = "TEST_BRAIN_GO_BRRR"
        test_value = "test_value"

        os.environ[test_var] = test_value
        assert os.environ.get(test_var) == test_value

        # Clean up
        del os.environ[test_var]
