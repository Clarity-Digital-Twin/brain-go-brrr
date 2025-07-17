"""Test environment variable configuration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestEnvironmentConfig:
    """Test environment variable configuration."""
    
    def test_eegpt_model_path_from_env(self):
        """Test that EEGPT_MODEL_PATH can be set from environment."""
        test_path = "/custom/path/to/model.ckpt"
        
        with patch.dict(os.environ, {"EEGPT_MODEL_PATH": test_path}):
            # Re-import to pick up env var
            from api.main import EEGPT_MODEL_PATH
            
            assert str(EEGPT_MODEL_PATH).endswith("model.ckpt")
            
    def test_eegpt_model_path_default(self):
        """Test that EEGPT_MODEL_PATH uses default when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if exists
            os.environ.pop("EEGPT_MODEL_PATH", None)
            
            # Re-import module
            import importlib
            import api.main
            importlib.reload(api.main)
            
            from api.main import EEGPT_MODEL_PATH
            
            # Should use default path
            assert "pretrained/eegpt_mcae_58chs_4s_large4E.ckpt" in str(EEGPT_MODEL_PATH)
            
    def test_model_path_is_absolute(self):
        """Test that model path is always absolute."""
        with patch.dict(os.environ, {"EEGPT_MODEL_PATH": "relative/path/model.ckpt"}):
            import importlib
            import api.main
            importlib.reload(api.main)
            
            from api.main import EEGPT_MODEL_PATH
            
            assert EEGPT_MODEL_PATH.is_absolute()