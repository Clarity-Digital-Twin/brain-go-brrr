"""Tests for configuration module."""

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
        assert model_config.device == "auto"
        assert model_config.batch_size == 8
        assert model_config.sampling_rate == 256
        assert model_config.window_duration == 4.0
        assert model_config.patch_size == 64
        assert model_config.n_summary_tokens == 4
        assert model_config.embed_dim == 512
        assert model_config.streaming_threshold == 120.0
        assert model_config.window_overlap == 0.5
        assert model_config.model_path.name == "eegpt_mcae_58chs_4s_large4E.ckpt"

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

    def test_model_config_invalid_window_duration(self):
        """Test that invalid window duration raises error."""
        import pytest

        # Window duration that doesn't result in integer samples at 256 Hz
        config = ModelConfig(window_duration=3.7)  # 3.7 * 256 = 947.2 (not integer)
        # The error happens when accessing the property
        with pytest.raises(ValueError, match="Window duration must result in integer samples"):
            _ = config.window_samples

    def test_model_config_missing_model_path(self, tmp_path):
        """Test that missing model path raises error."""
        import pytest

        nonexistent_path = tmp_path / "nonexistent_model.ckpt"
        with pytest.raises(ValueError, match="EEGPT model checkpoint not found"):
            ModelConfig(model_path=nonexistent_path)
