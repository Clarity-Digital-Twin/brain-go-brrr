"""Tests for configuration module."""

import pytest
from pathlib import Path

from brain_go_brrr.core.config import Config, ModelConfig, TrainingConfig


class TestConfig:
    """Test configuration management."""

    def test_default_config_creation(self):
        """Test creating a default configuration."""
        config = Config()
        assert config.environment == "development"
        assert config.debug is False
        assert config.seed == 42
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)

    def test_model_config_defaults(self):
        """Test default model configuration."""
        model_config = ModelConfig()
        assert model_config.name == "eegpt"
        assert model_config.embed_dim == 128
        assert model_config.num_heads == 8
        assert model_config.num_layers == 6
        assert model_config.patch_size == 64
        assert model_config.mask_ratio == 0.75
        assert model_config.dropout == 0.1

    def test_training_config_defaults(self):
        """Test default training configuration."""
        training_config = TrainingConfig()
        assert training_config.batch_size == 32
        assert training_config.learning_rate == 1e-4
        assert training_config.num_epochs == 100
        assert training_config.warmup_steps == 1000
        assert training_config.weight_decay == 0.01
        assert training_config.gradient_clip_norm == 1.0
        assert training_config.use_ddp is False
        assert training_config.num_gpus == 1

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        config = Config(
            environment="production",
            debug=True,
            seed=123,
        )
        assert config.environment == "production"
        assert config.debug is True
        assert config.seed == 123

    def test_directory_creation(self):
        """Test that necessary directories are created."""
        config = Config()
        # These should be created in model_post_init
        assert config.data_dir.exists()
        assert config.output_dir.exists()
        assert config.log_dir.exists()