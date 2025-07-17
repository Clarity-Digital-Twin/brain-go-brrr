"""Tests for CLI module."""

import pytest
from typer.testing import CliRunner

from brain_go_brrr.cli import app


class TestCLI:
    """Test command-line interface."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Brain Go Brrr version" in result.stdout

    def test_help_command(self):
        """Test help command."""
        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Brain Go Brrr" in result.stdout
        assert "train" in result.stdout
        assert "preprocess" in result.stdout
        assert "evaluate" in result.stdout
        assert "serve" in result.stdout

    def test_train_command_help(self):
        """Test train command help."""
        result = self.runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Train an EEGPT model" in result.stdout

    def test_preprocess_command_help(self):
        """Test preprocess command help."""
        result = self.runner.invoke(app, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "Preprocess EEG data" in result.stdout

    def test_evaluate_command_help(self):
        """Test evaluate command help."""
        result = self.runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "Evaluate a trained model" in result.stdout

    def test_serve_command_help(self):
        """Test serve command help."""
        result = self.runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "Serve a trained model" in result.stdout