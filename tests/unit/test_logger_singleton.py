"""Test logger singleton behavior."""

import logging
import tempfile
from pathlib import Path

from src.brain_go_brrr.core.logger import get_logger


class TestLoggerSingleton:
    """Test that loggers are properly singleton-ized."""

    def test_logger_handlers_not_duplicated(self):
        """Test that multiple calls to get_logger don't duplicate handlers."""
        # Get logger twice with same name
        logger1 = get_logger("test_singleton")
        initial_handler_count = len(logger1.handlers)

        logger2 = get_logger("test_singleton")

        # Should be same instance
        assert logger1 is logger2

        # Should not have added more handlers
        assert len(logger2.handlers) == initial_handler_count

    def test_logger_propagation_disabled(self):
        """Test that logger propagation is disabled to avoid duplicates."""
        logger = get_logger("test_propagation")
        assert logger.propagate is False

    def test_different_names_get_different_loggers(self):
        """Test that different names create different logger instances."""
        logger1 = get_logger("test_logger_1")
        logger2 = get_logger("test_logger_2")

        assert logger1 is not logger2
        assert logger1.name == "test_logger_1"
        assert logger2.name == "test_logger_2"

    def test_logger_level_setting(self):
        """Test that logger level is properly set."""
        logger_info = get_logger("test_level_info", level="INFO")
        logger_debug = get_logger("test_level_debug", level="DEBUG")

        assert logger_info.level == logging.INFO
        assert logger_debug.level == logging.DEBUG

    def test_repeated_imports_dont_duplicate_handlers(self):
        """Test that repeated module imports don't duplicate handlers."""
        # Simulate repeated imports
        for _ in range(3):
            from src.brain_go_brrr.core.logger import get_logger as get_logger_again

            logger = get_logger_again("test_import_duplication")

        # Should still have only one set of handlers
        assert len(logger.handlers) > 0  # Has handlers
        assert len(logger.handlers) <= 2  # At most console + file handler

    def test_logger_without_rich_console(self):
        """Test logger with standard console handler."""
        logger = get_logger("test_no_rich", level="DEBUG", rich_console=False)

        # Should have standard stream handler
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)

        # Check formatter is standard format
        formatter = handler.formatter
        assert formatter is not None
        assert "asctime" in formatter._fmt
        assert "levelname" in formatter._fmt

    def test_logger_with_file_handler(self):
        """Test logger with file handler."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test_logs" / "test.log"

            logger = get_logger(
                "test_file_handler", level="INFO", log_file=log_file, rich_console=True
            )

            # Should have 2 handlers: console + file
            assert len(logger.handlers) == 2

            # Find file handler
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 1

            # Check file was created
            assert log_file.exists()

            # Test logging to file
            logger.info("Test message to file")

            # Verify content was written
            with log_file.open() as f:
                content = f.read()
                assert "Test message to file" in content
                assert "INFO" in content

    def test_logger_with_all_options(self):
        """Test logger with all options enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "all_options.log"

            logger = get_logger(
                "test_all_options",
                level="WARNING",
                log_file=log_file,
                rich_console=False,  # Use standard console
            )

            # Should have 2 handlers
            assert len(logger.handlers) == 2

            # Check level
            assert logger.level == logging.WARNING

            # Log a warning
            logger.warning("Test warning message")

            # Check file content
            with log_file.open() as f:
                content = f.read()
                assert "Test warning message" in content
                assert "WARNING" in content

    def test_logger_creates_parent_directories(self):
        """Test that logger creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Nested path that doesn't exist
            log_file = Path(temp_dir) / "deep" / "nested" / "path" / "test.log"

            logger = get_logger("test_mkdir", log_file=log_file)

            # Parent directories should be created
            assert log_file.parent.exists()
            assert log_file.exists()

            # Test writing
            logger.info("Directory creation test")
            assert log_file.stat().st_size > 0
