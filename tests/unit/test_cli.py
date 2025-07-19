"""Tests for CLI module."""

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

    def test_stream_command_help(self):
        """Test stream command help."""
        result = self.runner.invoke(app, ["stream", "--help"])
        assert result.exit_code == 0
        assert "Stream EDF file" in result.stdout

    def test_train_command_without_args(self):
        """Test train command runs (even if not implemented)."""
        result = self.runner.invoke(app, ["train"])
        # Should run but show "not implemented yet"
        assert "Training logic not implemented yet" in result.stdout

    def test_train_command_with_debug(self):
        """Test train command with debug flag."""
        result = self.runner.invoke(app, ["train", "--debug"])
        assert "Training logic not implemented yet" in result.stdout

    def test_preprocess_command_with_args(self, tmp_path):
        """Test preprocess command runs (even if not implemented)."""
        # Create dummy paths
        data_path = tmp_path / "data.edf"
        output_path = tmp_path / "output.edf"
        data_path.touch()

        result = self.runner.invoke(app, ["preprocess", str(data_path), str(output_path)])
        # Should run but show "not implemented yet"
        assert "Preprocessing logic not implemented yet" in result.stdout

    def test_evaluate_command_with_args(self, tmp_path):
        """Test evaluate command runs (even if not implemented)."""
        # Create dummy paths
        model_path = tmp_path / "model.ckpt"
        data_path = tmp_path / "data.edf"
        model_path.touch()
        data_path.touch()

        result = self.runner.invoke(app, ["evaluate", str(model_path), str(data_path)])
        # Should run but show "not implemented yet"
        assert "logic not implemented yet" in result.stdout.lower()

    def test_serve_command_without_args(self):
        """Test serve command runs (even if not implemented)."""
        result = self.runner.invoke(app, ["serve"])
        # Should run but show "not implemented yet" or error about missing model
        assert result.exit_code != 0 or "not implemented" in result.stdout.lower()

    def test_version_command_shows_version(self):
        """Test version command shows actual version."""
        result = self.runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.stdout  # Should show version from pyproject.toml
