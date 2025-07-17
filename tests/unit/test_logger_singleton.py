"""Test logger singleton behavior."""

import logging

import pytest

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