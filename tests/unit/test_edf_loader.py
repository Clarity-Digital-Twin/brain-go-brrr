"""Tests for EDF loader - CLEAN, NO OVER-MOCKING."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestEdfLoader:
    """Test EDF loading functions."""

    @patch('brain_go_brrr.core.edf_loader.mne')
    def test_load_edf_safe_success(self, mock_mne):
        """Test successful EDF loading."""
        from brain_go_brrr.core.edf_loader import load_edf_safe

        # Setup mock
        mock_raw = MagicMock()
        mock_raw.info = {'sfreq': 256}
        mock_mne.io.read_raw_edf.return_value = mock_raw

        # Test
        result = load_edf_safe("test.edf")

        # Verify
        assert result == mock_raw
        mock_mne.io.read_raw_edf.assert_called_once()

    @patch('brain_go_brrr.core.edf_loader.mne')
    def test_load_edf_safe_file_not_found(self, mock_mne):
        """Test handling of missing file."""
        from brain_go_brrr.core.edf_loader import load_edf_safe
        from brain_go_brrr.core.exceptions import EdfLoadError

        # Setup mock to raise FileNotFoundError
        mock_mne.io.read_raw_edf.side_effect = FileNotFoundError("File not found")

        # Test
        with pytest.raises(EdfLoadError) as exc_info:
            load_edf_safe("missing.edf")

        assert "File not found" in str(exc_info.value)

    @patch('brain_go_brrr.core.edf_loader.mne')
    def test_load_edf_safe_invalid_format(self, mock_mne):
        """Test handling of invalid EDF format."""
        from brain_go_brrr.core.edf_loader import load_edf_safe
        from brain_go_brrr.core.exceptions import EdfLoadError

        # Setup mock to raise ValueError (invalid format)
        mock_mne.io.read_raw_edf.side_effect = ValueError("Invalid EDF header")

        # Test
        with pytest.raises(EdfLoadError) as exc_info:
            load_edf_safe("bad.edf")

        assert "Invalid EDF" in str(exc_info.value) or "header" in str(exc_info.value)

    def test_validate_edf_path_valid(self):
        """Test path validation for valid paths."""
        # Create a temp file
        import tempfile

        from brain_go_brrr.core.edf_loader import validate_edf_path
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Test - should not raise
            result = validate_edf_path(temp_path)
            assert result == temp_path

            # Test with string path
            result = validate_edf_path(str(temp_path))
            assert result == temp_path
        finally:
            temp_path.unlink()

    def test_validate_edf_path_invalid(self):
        """Test path validation for invalid paths."""
        from brain_go_brrr.core.edf_loader import validate_edf_path
        from brain_go_brrr.core.exceptions import EdfLoadError

        # Test non-existent file
        with pytest.raises(EdfLoadError) as exc_info:
            validate_edf_path("nonexistent.edf")
        assert "does not exist" in str(exc_info.value)

        # Test wrong extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(EdfLoadError) as exc_info:
                validate_edf_path(temp_path)
            assert "must be .edf" in str(exc_info.value) or "EDF file" in str(exc_info.value)
        finally:
            temp_path.unlink()

    @patch('brain_go_brrr.core.edf_loader.mne')
    def test_load_with_kwargs(self, mock_mne):
        """Test that kwargs are passed through correctly."""
        from brain_go_brrr.core.edf_loader import load_edf_safe

        mock_raw = MagicMock()
        mock_mne.io.read_raw_edf.return_value = mock_raw

        # Test with various kwargs
        load_edf_safe("test.edf", preload=True, verbose=False, stim_channel=None)

        # Verify kwargs were passed
        call_args = mock_mne.io.read_raw_edf.call_args
        assert call_args[1].get('preload') is True
        assert call_args[1].get('verbose') is False
        assert call_args[1].get('stim_channel') is None
