"""REAL tests for CLI commands - Clean, minimal mocking."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from brain_go_brrr.cli import (
    app,
    train,
    preprocess,
    evaluate,
    serve,
    stream,
)


class TestTrainCommand:
    """Test train command functionality."""
    
    @patch('brain_go_brrr.cli.Path')
    @patch('brain_go_brrr.cli.logger')
    def test_train_with_valid_config(self, mock_logger, mock_path):
        """Test train command with valid config."""
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_file.return_value = True
        
        # Should not raise
        with patch('brain_go_brrr.cli.run_training') as mock_run:
            train(
                config="config.yaml",
                data_dir="data",
                output_dir="output",
                epochs=10,
                batch_size=32,
                learning_rate=0.001,
                device="cpu",
                seed=42,
                debug=False
            )
            
            mock_run.assert_called_once()
            mock_logger.info.assert_called()
    
    def test_train_validates_config_exists(self):
        """Test train validates config file exists."""
        with pytest.raises(FileNotFoundError):
            train(
                config="nonexistent.yaml",
                data_dir="data",
                output_dir="output"
            )


class TestPreprocessCommand:
    """Test preprocess command functionality."""
    
    @patch('brain_go_brrr.cli.Path')
    @patch('brain_go_brrr.cli.logger')
    def test_preprocess_single_file(self, mock_logger, mock_path):
        """Test preprocessing single EDF file."""
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_file.return_value = True
        mock_path.return_value.suffix = ".edf"
        
        with patch('brain_go_brrr.cli.preprocess_edf') as mock_proc:
            preprocess(
                input_path="test.edf",
                output_dir="output",
                sampling_rate=256,
                low_freq=0.5,
                high_freq=50.0,
                notch_freq=60.0,
                normalize=True,
                debug=False
            )
            
            mock_proc.assert_called_once()
    
    @patch('brain_go_brrr.cli.Path')
    def test_preprocess_directory(self, mock_path):
        """Test preprocessing directory of files."""
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_dir.return_value = True
        
        # Mock glob to return EDF files
        mock_path.return_value.glob.return_value = [
            Path("file1.edf"),
            Path("file2.edf")
        ]
        
        with patch('brain_go_brrr.cli.preprocess_edf') as mock_proc:
            preprocess(
                input_path="data_dir",
                output_dir="output"
            )
            
            assert mock_proc.call_count == 2


class TestEvaluateCommand:
    """Test evaluate command functionality."""
    
    @patch('brain_go_brrr.cli.Path')
    @patch('brain_go_brrr.cli.load_model')
    @patch('brain_go_brrr.cli.load_test_data')
    def test_evaluate_model(self, mock_data, mock_model, mock_path):
        """Test model evaluation."""
        mock_path.return_value.exists.return_value = True
        mock_model.return_value = MagicMock()
        mock_data.return_value = (MagicMock(), MagicMock())
        
        with patch('brain_go_brrr.cli.run_evaluation') as mock_eval:
            mock_eval.return_value = {"accuracy": 0.95}
            
            evaluate(
                model_path="model.ckpt",
                data_dir="test_data",
                output_dir="results",
                batch_size=32,
                device="cpu",
                debug=False
            )
            
            mock_eval.assert_called_once()
    
    def test_evaluate_validates_model_exists(self):
        """Test evaluate validates model file exists."""
        with pytest.raises(FileNotFoundError):
            evaluate(
                model_path="nonexistent.ckpt",
                data_dir="data"
            )


class TestServeCommand:
    """Test serve command functionality."""
    
    @patch('brain_go_brrr.cli.uvicorn')
    def test_serve_starts_api(self, mock_uvicorn):
        """Test serve starts the API server."""
        serve(
            host="127.0.0.1",
            port=8000,
            reload=False,
            workers=1,
            debug=False
        )
        
        mock_uvicorn.run.assert_called_once_with(
            "brain_go_brrr.api.app:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            workers=1,
            log_level="info"
        )
    
    @patch('brain_go_brrr.cli.uvicorn')
    def test_serve_debug_mode(self, mock_uvicorn):
        """Test serve in debug mode."""
        serve(
            host="0.0.0.0",
            port=8080,
            reload=True,
            workers=1,
            debug=True
        )
        
        mock_uvicorn.run.assert_called_once_with(
            "brain_go_brrr.api.app:app",
            host="0.0.0.0",
            port=8080,
            reload=True,
            workers=1,
            log_level="debug"
        )


class TestStreamCommand:
    """Test stream command functionality."""
    
    @patch('brain_go_brrr.cli.StreamProcessor')
    def test_stream_tcp(self, mock_processor):
        """Test TCP streaming mode."""
        mock_proc_instance = MagicMock()
        mock_processor.return_value = mock_proc_instance
        
        stream(
            mode="tcp",
            host="localhost",
            port=5555,
            channels=20,
            sampling_rate=256,
            buffer_size=1024,
            debug=False
        )
        
        mock_processor.assert_called_once_with(
            mode="tcp",
            host="localhost",
            port=5555,
            channels=20,
            sampling_rate=256,
            buffer_size=1024
        )
        mock_proc_instance.start.assert_called_once()
    
    @patch('brain_go_brrr.cli.StreamProcessor')
    def test_stream_lsl(self, mock_processor):
        """Test LSL streaming mode."""
        mock_proc_instance = MagicMock()
        mock_processor.return_value = mock_proc_instance
        
        stream(
            mode="lsl",
            stream_name="EEG",
            channels=32,
            sampling_rate=500,
            buffer_size=2048,
            debug=True
        )
        
        mock_processor.assert_called_once()
        mock_proc_instance.start.assert_called_once()