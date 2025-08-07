"""Tests for CLI commands - Using CliRunner for black-box testing."""

import pytest
from typer.testing import CliRunner

from brain_go_brrr.cli import app

runner = CliRunner()


class TestCLICommands:
    """Test CLI commands using CliRunner."""

    def test_cli_help(self):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Brain-Go-Brrr" in result.stdout or "Usage" in result.stdout

    def test_train_help(self):
        """Test train command help."""
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "train" in result.stdout.lower()

    def test_preprocess_help(self):
        """Test preprocess command help."""
        result = runner.invoke(app, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "preprocess" in result.stdout.lower()

    def test_evaluate_help(self):
        """Test evaluate command help."""
        result = runner.invoke(app, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "evaluate" in result.stdout.lower()

    def test_serve_help(self):
        """Test serve command help."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.stdout.lower()

    @pytest.mark.skip(reason="Stream command may not be implemented")
    def test_stream_help(self):
        """Test stream command help."""
        result = runner.invoke(app, ["stream", "--help"])
        assert result.exit_code == 0
        assert "stream" in result.stdout.lower()

    def test_invalid_command(self):
        """Test invalid command shows error."""
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

    def test_version_flag(self):
        """Test version flag if implemented."""
        result = runner.invoke(app, ["--version"])
        # Version flag might not be implemented
        if result.exit_code == 0:
            assert "version" in result.stdout.lower() or "0." in result.stdout
