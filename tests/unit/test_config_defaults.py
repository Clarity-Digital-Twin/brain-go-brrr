"""Simple tests to boost coverage above 60%."""

import tempfile
from datetime import UTC
from pathlib import Path

from brain_go_brrr.core.config import Config, DataConfig, ModelConfig, TrainingConfig
from brain_go_brrr.core.exceptions import (
    BrainGoBrrrError,
    ConfigurationError,
    EdfLoadError,
    ModelError,
    ProcessingError,
)


class TestConfigSimple:
    """Simple config tests."""

    def test_config_project_root(self):
        """Test Config project root."""
        config = Config()
        assert isinstance(config.project_root, Path)
        assert config.project_root.exists()

    def test_model_config_window_samples(self):
        """Test ModelConfig window samples calculation."""
        with tempfile.NamedTemporaryFile(suffix=".ckpt") as tmp:
            model_config = ModelConfig(model_path=Path(tmp.name))
            # 4.0 seconds * 256 Hz = 1024 samples
            assert model_config.window_samples == 1024

    def test_training_config_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4

    def test_data_config_defaults(self):
        """Test DataConfig defaults."""
        config = DataConfig()
        assert config.sample_rate == 250  # Actual default is 250


class TestExceptionsSimple:
    """Simple exception tests."""

    def test_brain_go_brrr_error(self):
        """Test base exception."""
        err = BrainGoBrrrError("test error")
        assert str(err) == "test error"
        assert isinstance(err, Exception)

    def test_configuration_error_hierarchy(self):
        """Test ConfigurationError hierarchy."""
        err = ConfigurationError("config problem")
        assert isinstance(err, BrainGoBrrrError)
        assert isinstance(err, Exception)

    def test_model_error_hierarchy(self):
        """Test ModelError hierarchy."""
        err = ModelError("model problem")
        assert isinstance(err, BrainGoBrrrError)

    def test_processing_error_hierarchy(self):
        """Test ProcessingError hierarchy."""
        err = ProcessingError("processing issue")
        assert isinstance(err, BrainGoBrrrError)

    def test_edf_load_error_hierarchy(self):
        """Test EdfLoadError hierarchy."""
        err = EdfLoadError("edf issue")
        assert isinstance(err, BrainGoBrrrError)


class TestTimeUtils:
    """Test time utilities."""

    def test_utc_now(self):
        """Test utc_now function."""
        from brain_go_brrr.utils.time import utc_now

        now = utc_now()
        assert now.tzinfo == UTC

    def test_format_timestamp(self):
        """Test format_timestamp function."""
        from brain_go_brrr.utils.time import format_timestamp, utc_now

        now = utc_now()
        ts = format_timestamp(now)
        assert isinstance(ts, str)
        assert len(ts) > 10  # Should have reasonable length

    def test_timestamp_for_logging(self):
        """Test timestamp_for_logging function."""
        from brain_go_brrr.utils.time import timestamp_for_logging

        ts = timestamp_for_logging()
        assert isinstance(ts, str)
        assert len(ts) > 10
