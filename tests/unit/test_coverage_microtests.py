"""Microtests for coverage boost - targeting real, simple functionality."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tests.fakes import FakeMNERaw


class TestAPIEndpoints:
    """Test basic API functionality."""
    
    def test_health_check_returns_200(self):
        """Test health check endpoint returns expected structure."""
        from brain_go_brrr.api.routers.health import health_check
        
        result = asyncio.run(health_check())
        
        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "service" in result
        assert result["service"] == "brain-go-brrr-api"
    
    def test_readiness_check_returns_ready(self):
        """Test readiness endpoint structure."""
        from brain_go_brrr.api.routers.health import readiness_check
        
        result = asyncio.run(readiness_check())
        
        assert result["status"] == "ready"
        assert "timestamp" in result


class TestEnums:
    """Test enum values are defined correctly."""
    
    def test_job_status_enum_values(self):
        """Test JobStatus enum has expected values."""
        from brain_go_brrr.api.schemas import JobStatus
        
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
    
    def test_job_priority_enum_values(self):
        """Test JobPriority enum has expected values."""
        from brain_go_brrr.api.schemas import JobPriority
        
        assert JobPriority.LOW.value == "low"
        assert JobPriority.NORMAL.value == "normal"
        assert JobPriority.HIGH.value == "high"
    
    def test_triage_level_enum_values(self):
        """Test TriageLevel enum values."""
        from brain_go_brrr.core.abnormal.detector import TriageLevel
        
        assert TriageLevel.NORMAL.value == "NORMAL"
        assert TriageLevel.ROUTINE.value == "ROUTINE"
        assert TriageLevel.EXPEDITE.value == "EXPEDITE"
        assert TriageLevel.URGENT.value == "URGENT"


class TestDataStructures:
    """Test data structures and configs."""
    
    def test_eegpt_config_computed_properties(self):
        """Test EEGPT config calculates properties correctly."""
        from brain_go_brrr.models.eegpt_model import EEGPTConfig
        
        config = EEGPTConfig(window_duration=4.0, sampling_rate=256)
        
        assert config.window_samples == 1024  # 4 * 256
        assert config.n_patches_per_window == 16  # 1024 / 64
    
    def test_model_config_has_defaults(self):
        """Test ModelConfig has sensible defaults."""
        from brain_go_brrr.core.config import ModelConfig
        
        config = ModelConfig()
        
        assert config.device in ["cpu", "cuda", "auto"]
        assert config.batch_size > 0
        assert config.sampling_rate == 256
        assert config.window_duration == 4.0
    
    def test_abnormality_config_structure(self):
        """Test AbnormalityConfig has expected structure."""
        from brain_go_brrr.core.abnormality_config import AbnormalityConfig
        
        config = AbnormalityConfig()
        
        assert hasattr(config, "classification")
        assert hasattr(config, "quality")
        assert hasattr(config, "processing")
        assert hasattr(config, "model")


class TestWindowExtractor:
    """Test window extraction edge cases."""
    
    def test_window_extractor_single_window(self):
        """Test extraction with exactly one window."""
        from brain_go_brrr.core.window_extractor import WindowExtractor
        
        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=0.0)
        
        # Create data for exactly one window
        sfreq = 256
        data = np.random.randn(19, 4 * sfreq) * 1e-6
        
        windows = extractor.extract(data, sfreq)
        
        assert len(windows) == 1
        assert windows[0].shape == (19, 4 * sfreq)
    
    def test_window_extractor_with_overlap(self):
        """Test extraction with overlapping windows."""
        from brain_go_brrr.core.window_extractor import WindowExtractor
        
        extractor = WindowExtractor(window_seconds=4.0, overlap_seconds=2.0)
        
        # Create 10s of data
        sfreq = 256
        data = np.random.randn(19, 10 * sfreq) * 1e-6
        
        windows = extractor.extract(data, sfreq)
        
        # With 4s windows, 2s overlap (2s step): (10-4)/2 + 1 = 4 windows
        assert len(windows) == 4
        assert all(w.shape == (19, 4 * sfreq) for w in windows)


class TestConfigOverrides:
    """Test configuration with overrides."""
    
    def test_eegpt_model_config_override(self):
        """Test EEGPTModel config can be overridden."""
        from brain_go_brrr.models.eegpt_model import EEGPTConfig
        
        # Override defaults
        config = EEGPTConfig(
            window_duration=8.0,
            sampling_rate=512,
            n_channels=32
        )
        
        assert config.window_duration == 8.0
        assert config.sampling_rate == 512
        assert config.n_channels == 32
        assert config.window_samples == 8 * 512  # Computed property updates
    
    def test_model_config_device_selection(self):
        """Test device selection logic."""
        from brain_go_brrr.core.config import ModelConfig
        
        # Test explicit device
        config = ModelConfig(device="cpu")
        assert config.device == "cpu"
        
        # Test auto device - should pick cuda if available, else cpu
        config_auto = ModelConfig(device="auto")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config_auto.device in ["auto", expected]  # May stay as "auto"


class TestEdgeValidation:
    """Test edge cases and validation."""
    
    def test_validation_result_structure(self):
        """Test ValidationResult dataclass works."""
        from brain_go_brrr.core.edf_validator import ValidationResult
        
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Channel count low"],
            metadata={"channels": 16}
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.metadata["channels"] == 16
    
    def test_snippet_maker_initialization(self):
        """Test EEGSnippetMaker initialization."""
        from brain_go_brrr.core.snippets.maker import EEGSnippetMaker
        
        maker = EEGSnippetMaker(snippet_length=2.0, overlap=0.5)
        
        assert maker.snippet_length == 2.0
        assert maker.overlap == 0.5
    
    def test_edf_validator_params(self):
        """Test EDFValidator parameter handling."""
        from brain_go_brrr.core.edf_validator import EDFValidator
        
        validator = EDFValidator(
            min_duration_seconds=60.0,
            min_channels=19,
            max_amplitude_v=1e-3
        )
        
        assert validator.min_duration_seconds == 60.0
        assert validator.min_channels == 19
        assert validator.max_amplitude_v == 1e-3


class TestArchitectureComponents:
    """Test core architecture components."""
    
    def test_rotary_embedding_initialization(self):
        """Test RoPE initialization."""
        from brain_go_brrr.models.eegpt_architecture import RoPE
        
        rope = RoPE(dim=64, theta=10000.0, max_seq_len=1024)
        
        assert rope.dim == 64
        assert rope.theta == 10000.0
        assert rope.max_seq_len == 1024
    
    def test_eeg_transformer_structure(self):
        """Test EEGTransformer has expected structure."""
        from brain_go_brrr.models.eegpt_architecture import EEGTransformer
        
        model = EEGTransformer(
            n_channels=["Fp1", "Fp2", "C3", "C4"],
            patch_size=64,
            embed_dim=256,
            depth=4,
            num_heads=4
        )
        
        assert len(model.n_channels) == 4
        assert model.patch_size == 64
        assert model.embed_dim == 256
        assert len(model.blocks) == 4  # Depth = number of transformer blocks