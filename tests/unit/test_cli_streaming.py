"""Integration tests for CLI streaming functionality.

Tests the end-to-end streaming workflow:
1. CLI invocation with EDF file
2. Streaming processing with windowing
3. JSON output format validation
4. Error handling for invalid inputs
"""

import contextlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sample_edf_path(project_root) -> Path:
    """Get path to a sample EDF file for testing."""
    edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"
    if not edf_path.exists():
        pytest.skip("Sleep-EDF data not available for integration tests")
    return edf_path


@pytest.fixture
def short_edf_path(sleep_edf_path) -> Path:
    """Use existing EDF file for testing."""
    # For now, use the full file but tests will only process first few windows
    return sleep_edf_path


class TestCLIStreamingIntegration:
    """Test CLI streaming functionality end-to-end."""

    def run_cli_command(self, args: list[str]) -> subprocess.CompletedProcess:
        """Run CLI command and return result."""
        cmd = [sys.executable, "-m", "brain_go_brrr.cli", *args]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )
        return result

    def parse_json_output(self, output: str) -> list[dict[str, Any]]:
        """Parse JSON lines from CLI output."""
        lines = output.strip().split("\n")
        results = []

        for line in lines:
            if line.strip() and line.startswith("{"):
                import contextlib

                with contextlib.suppress(json.JSONDecodeError):
                    results.append(json.loads(line))

        return results

    @pytest.mark.integration
    def test_stream_basic_functionality(self, short_edf_path):
        """Test basic streaming with default parameters."""
        # Run streaming command with limited windows for speed
        result = self.run_cli_command(
            [
                "stream",
                str(short_edf_path),
                "--format",
                "json",
                "--max-windows",
                "5",  # Only process 5 windows for testing
            ]
        )

        # Check command succeeded
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Parse JSON output
        windows = self.parse_json_output(result.stdout)

        # Verify we got windows
        assert len(windows) > 0, "No windows processed"

        # Verify window structure
        first_window = windows[0]
        assert "window" in first_window
        assert "start_time" in first_window
        assert "end_time" in first_window
        assert "feature_shape" in first_window
        assert "feature_mean" in first_window
        assert "feature_std" in first_window

        # Verify window numbering
        for i, window in enumerate(windows):
            assert window["window"] == i + 1

        # Verify time progression
        for i in range(1, len(windows)):
            assert windows[i]["start_time"] >= windows[i - 1]["end_time"]

    @pytest.mark.integration
    def test_stream_with_custom_window_size(self, short_edf_path):
        """Test streaming with custom window size."""
        window_size = 2.0  # 2 second windows

        result = self.run_cli_command(
            [
                "stream",
                str(short_edf_path),
                "--window-size",
                str(window_size),
                "--format",
                "json",
                "--max-windows",
                "10",
            ]
        )

        assert result.returncode == 0
        windows = self.parse_json_output(result.stdout)

        # Verify window durations
        for window in windows:
            duration = window["end_time"] - window["start_time"]
            assert abs(duration - window_size) < 0.01  # Allow small floating point error

    @pytest.mark.integration
    def test_stream_with_overlap(self, short_edf_path):
        """Test streaming with overlapping windows."""
        window_size = 4.0
        overlap = 0.5  # 50% overlap

        result = self.run_cli_command(
            [
                "stream",
                str(short_edf_path),
                "--window-size",
                str(window_size),
                "--overlap",
                str(overlap),
                "--format",
                "json",
                "--max-windows",
                "10",
            ]
        )

        assert result.returncode == 0
        windows = self.parse_json_output(result.stdout)

        # Verify overlap
        if len(windows) > 1:
            expected_step = window_size * (1 - overlap)
            for i in range(1, len(windows)):
                actual_step = windows[i]["start_time"] - windows[i - 1]["start_time"]
                assert abs(actual_step - expected_step) < 0.01

    @pytest.mark.integration
    def test_stream_feature_extraction(self, short_edf_path):
        """Test that features are actually extracted."""
        result = self.run_cli_command(
            [
                "stream",
                str(short_edf_path),
                "--window-size",
                "4.0",
                "--format",
                "json",
                "--max-windows",
                "5",
            ]
        )

        assert result.returncode == 0
        windows = self.parse_json_output(result.stdout)

        for window in windows:
            # Check feature shape is correct
            feature_shape = window["feature_shape"]
            assert len(feature_shape) == 2  # Should be 2D
            assert feature_shape[0] > 0  # Should have features
            assert feature_shape[1] == 512  # EEGPT embedding dimension

            # Check statistics are reasonable
            assert isinstance(window["feature_mean"], float)
            assert isinstance(window["feature_std"], float)
            assert window["feature_std"] > 0  # Should have variation

    @pytest.mark.integration
    def test_stream_invalid_file(self, tmp_path):
        """Test error handling for invalid file."""
        invalid_path = tmp_path / "nonexistent.edf"

        result = self.run_cli_command(["stream", str(invalid_path)])

        # Should fail with non-zero exit code
        assert result.returncode != 0
        assert "not found" in result.stderr or "not found" in result.stdout

    @pytest.mark.integration
    def test_stream_invalid_overlap(self, short_edf_path):
        """Test error handling for invalid overlap values."""
        result = self.run_cli_command(
            [
                "stream",
                str(short_edf_path),
                "--overlap",
                "1.5",  # Invalid: > 1.0
            ]
        )

        # Should fail
        assert result.returncode != 0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_stream_large_file_performance(self, sample_edf_path):
        """Test streaming performance with large file."""
        import time

        start_time = time.time()

        # Stream first 10 windows only
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "brain_go_brrr.cli",
                "stream",
                str(sample_edf_path),
                "--format",
                "json",
                "--max-windows",
                "10",
            ],
            capture_output=True,
            text=True,
            timeout=30,  # Should complete quickly with streaming
        )

        elapsed = time.time() - start_time

        assert result.returncode == 0

        # Parse at least first few windows
        lines = result.stdout.strip().split("\n")[:10]  # First 10 windows
        windows = []
        for line in lines:
            if line.strip() and line.startswith("{"):
                with contextlib.suppress(json.JSONDecodeError):
                    windows.append(json.loads(line))

        assert len(windows) >= 5  # Should process at least 5 windows

        # Should stream efficiently (not load entire file)
        assert elapsed < 10.0  # Should complete in under 10 seconds

    @pytest.mark.integration
    def test_stream_output_ordering(self, short_edf_path):
        """Test that output windows are in correct temporal order."""
        result = self.run_cli_command(
            [
                "stream",
                str(short_edf_path),
                "--window-size",
                "2.0",
                "--format",
                "json",
                "--max-windows",
                "10",
            ]
        )

        assert result.returncode == 0
        windows = self.parse_json_output(result.stdout)

        # Verify temporal ordering
        for i in range(1, len(windows)):
            assert windows[i]["start_time"] > windows[i - 1]["start_time"]
            assert windows[i]["window"] == windows[i - 1]["window"] + 1

    @pytest.mark.integration
    def test_stream_csv_format(self, short_edf_path):
        """Test non-JSON output format."""
        result = self.run_cli_command(
            [
                "stream",
                str(short_edf_path),
                "--window-size",
                "4.0",
                "--format",
                "csv",
                "--max-windows",
                "5",
            ]
        )

        assert result.returncode == 0

        # Should have human-readable output
        output_lines = result.stdout.strip().split("\n")
        window_lines = [line for line in output_lines if line.startswith("Window")]

        assert len(window_lines) > 0

        # Check format
        for line in window_lines:
            assert "Window" in line
            assert " - " in line  # Time range separator


class TestCLIStreamingEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.integration
    def test_stream_empty_file(self, tmp_path):
        """Test handling of empty or corrupt EDF file."""
        # Create an empty file with .edf extension
        empty_edf = tmp_path / "empty.edf"
        empty_edf.write_text("")

        result = subprocess.run(
            [sys.executable, "-m", "brain_go_brrr.cli", "stream", str(empty_edf)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should fail gracefully
        assert result.returncode != 0

    @pytest.mark.integration
    def test_stream_keyboard_interrupt(self, short_edf_path):
        """Test graceful handling of interruption."""
        # This is hard to test automatically, but we can at least
        # verify the command starts successfully and handles termination gracefully
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "brain_go_brrr.cli",
                "stream",
                str(short_edf_path),
                "--format",
                "json",
                "--max-windows",
                "2",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Let it run briefly to ensure it starts processing
        import time

        time.sleep(0.5)

        # Terminate it (simulating Ctrl+C)
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()

        # Should have either exited cleanly or been terminated
        # The important thing is it doesn't hang indefinitely
        assert proc.returncode is not None, "Process did not terminate"

        # Verify the process behavior based on return code
        if proc.returncode == -15:  # SIGTERM (Unix)
            # Process was terminated as expected
            # Check if it started processing before termination
            if stdout:
                # Should have valid JSON output if it produced any
                lines = stdout.strip().split("\n")
                for line in lines:
                    if line.strip() and line.startswith("{"):
                        # Verify it's valid JSON
                        import json

                        try:
                            json.loads(line)
                        except json.JSONDecodeError:
                            pytest.fail(f"Invalid JSON output: {line}")
        elif proc.returncode == 1 and sys.platform == "win32":  # Windows termination
            # Windows uses different signal codes
            pass
        elif proc.returncode != 0:
            # If there was an error, it should be in stderr
            assert stderr, f"Process failed with code {proc.returncode} but no error output"
            # Log the error for debugging
        else:
            # Process completed normally (max-windows=2 is small)
            assert stdout, "Process completed but produced no output"
            # Verify we got valid streaming output
            windows = self.parse_json_output(stdout)
            assert len(windows) >= 1, "Should have processed at least one window"


@pytest.mark.integration
class TestCLIStreamingIntegrationWithModel:
    """Test streaming with actual model checkpoint if available."""

    @pytest.fixture
    def model_checkpoint_path(self, project_root) -> Path:
        """Get model checkpoint path if available."""
        checkpoint = project_root / "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
        if not checkpoint.exists():
            pytest.skip("Model checkpoint not available")
        return checkpoint

    def test_stream_with_real_model(self, short_edf_path, model_checkpoint_path, monkeypatch):
        """Test streaming with real model weights."""
        # Set environment variable for model path
        monkeypatch.setenv("EEGPT_CHECKPOINT_PATH", str(model_checkpoint_path))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "brain_go_brrr.cli",
                "stream",
                str(short_edf_path),
                "--window-size",
                "4.0",
                "--format",
                "json",
                "--max-windows",
                "5",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0

        # Parse output
        windows = []
        for line in result.stdout.strip().split("\n"):
            if line.strip() and line.startswith("{"):
                with contextlib.suppress(json.JSONDecodeError):
                    windows.append(json.loads(line))

        assert len(windows) > 0

        # Features from real model should have specific properties
        for window in windows:
            # Real features should have meaningful variation
            assert window["feature_std"] > 0.01  # Not near zero
            assert abs(window["feature_mean"]) < 10  # Reasonable range
