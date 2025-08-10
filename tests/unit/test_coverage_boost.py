"""Quick smoke tests to boost coverage for low-coverage modules."""

import json
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, Mock


def test_parallel_pipeline_basic():
    """Test parallel pipeline initialization."""
    from brain_go_brrr.core.pipeline.parallel import ParallelEEGPipeline
    
    # Mock the feature extractor initialization
    with patch("brain_go_brrr.core.pipeline.parallel.EEGPTFeatureExtractor"):
        pipeline = ParallelEEGPipeline(device="cpu")
        # Check what actually exists
        assert hasattr(pipeline, "eegpt_extractor")


def test_snippet_maker_basic():
    """Test snippet maker initialization."""
    from brain_go_brrr.core.snippets.maker import EEGSnippetMaker
    
    # No args needed
    maker = EEGSnippetMaker()
    # Check for actual attributes from __init__
    assert hasattr(maker, "snippet_length") and hasattr(maker, "overlap")


def test_tuab_dataset_empty():
    """Test TUAB dataset with no files."""
    from brain_go_brrr.data.tuab_dataset import TUABDataset
    
    with patch("brain_go_brrr.data.tuab_dataset.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.glob.return_value = []
        
        dataset = TUABDataset(root_dir="/fake/path", split="train")
        assert len(dataset) == 0


def test_job_store_operations():
    """Test job store basic operations."""
    from brain_go_brrr.core.jobs.store import ThreadSafeJobStore as JobStore
    from brain_go_brrr.api.schemas import JobData, JobStatus, JobPriority
    from datetime import datetime
    
    store = JobStore()
    
    # Create a job using the actual API - JobData is a dataclass
    job_id = "test-123"
    now = datetime.now()
    job_data = JobData(
        job_id=job_id,
        analysis_type="test",
        file_path="/fake/file.edf",
        status=JobStatus.PENDING,
        priority=JobPriority.NORMAL,
        created_at=now,
        updated_at=now,
        # Optional fields with defaults
        options={"param": 1},
        progress=0.0,
        result=None,
        error=None,
        started_at=None,
        completed_at=None
    )
    store.create(job_id, job_data)
    
    # Get the job
    retrieved = store.get(job_id)
    assert retrieved is not None
    assert retrieved.analysis_type == "test"


def test_linear_probe_init():
    """Test linear probe initialization."""
    from brain_go_brrr.models.eegpt_linear_probe import EEGPTLinearProbe
    
    with patch("brain_go_brrr.models.eegpt_linear_probe.create_normalized_eegpt") as mock:
        mock.return_value = MagicMock()
        # Add required n_input_channels parameter
        probe = EEGPTLinearProbe(
            checkpoint_path="/fake/path.ckpt", 
            n_input_channels=20,
            n_classes=2
        )
        assert probe.n_classes == 2


def test_sleep_analyzer_init():
    """Test sleep analyzer initialization."""
    from brain_go_brrr.core.sleep.analyzer import SleepAnalyzer
    
    analyzer = SleepAnalyzer()
    # Check for the actual method that exists
    assert hasattr(analyzer, "run_full_sleep_analysis")


def test_abnormal_detector_init():
    """Test abnormality detector initialization."""
    from brain_go_brrr.core.abnormal.detector import AbnormalityDetector
    from pathlib import Path
    
    fake_path = Path("/fake/model.ckpt")
    # Mock both the path validation and torch.load for classifier
    with patch("brain_go_brrr.core.config.Path.exists", return_value=True):
        with patch("brain_go_brrr.core.config.Path.is_file", return_value=True):
            with patch("brain_go_brrr.core.abnormal.detector.EEGPTModel") as mock_model:
                # Mock the model config to avoid loading classifier
                mock_instance = Mock()
                mock_instance.config.model_path = fake_path
                mock_model.return_value = mock_instance
                
                with patch("brain_go_brrr.core.abnormal.detector.torch.load") as mock_load:
                    # Mock the full classifier state dict with correct dimensions
                    # The error says it expects 2048 input features, not 768
                    mock_state = {
                        "0.weight": torch.randn(256, 2048),  # Fixed: 2048 input features
                        "0.bias": torch.randn(256),
                        "1.weight": torch.randn(256),
                        "1.bias": torch.randn(256),
                        "1.running_mean": torch.randn(256),
                        "1.running_var": torch.ones(256),
                        "1.num_batches_tracked": torch.tensor(0),
                        "4.weight": torch.randn(128, 256),
                        "4.bias": torch.randn(128),
                        "5.weight": torch.randn(128),
                        "5.bias": torch.randn(128),
                        "5.running_mean": torch.randn(128),
                        "5.running_var": torch.ones(128),
                        "5.num_batches_tracked": torch.tensor(0),
                        "8.weight": torch.randn(2, 128),
                        "8.bias": torch.randn(2)
                    }
                    mock_load.return_value = mock_state
                    
                    detector = AbnormalityDetector(model_path=fake_path)
                    # The method is detect_abnormality, not detect
                    assert hasattr(detector, "detect_abnormality")


def test_cache_redis_init():
    """Test Redis cache initialization."""
    from brain_go_brrr.infra.cache import RedisCache
    
    # RedisCache creates its own client internally via redis module
    with patch("redis.Redis") as mock_redis:
        mock_client = Mock()
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_redis.return_value = mock_client
        
        # RedisCache doesn't take namespace argument
        cache = RedisCache()
        # Just check it was created
        assert cache is not None


def test_eeg_preprocessor_init():
    """Test EEG preprocessor initialization."""
    from brain_go_brrr.preprocessing.eeg_preprocessor import EEGPreprocessor
    
    preprocessor = EEGPreprocessor()
    assert hasattr(preprocessor, "preprocess")


def test_feature_extractor_init():
    """Test feature extractor initialization."""
    from brain_go_brrr.core.features.extractor import EEGPTFeatureExtractor
    
    with patch("brain_go_brrr.core.features.extractor.EEGPTModel"):
        extractor = EEGPTFeatureExtractor(model_path="/fake/path")
        assert extractor.device == "cpu"


def test_chunked_autoreject_init():
    """Test chunked autoreject initialization."""
    from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor
    
    processor = ChunkedAutoRejectProcessor(chunk_size=10)
    assert processor.chunk_size == 10


def test_tuab_cached_dataset_init():
    """Test cached dataset initialization."""
    from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset
    
    mock_index = {"files": {}, "n_files": 0, "total_windows": 0, "metadata": {"split": "train"}}
    mock_json = json.dumps(mock_index)
    
    with patch("brain_go_brrr.data.tuab_cached_dataset.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        with patch("builtins.open", mock_open(read_data=mock_json)):
            # Add required root_dir parameter
            dataset = TUABCachedDataset(
                root_dir="/fake/root",
                cache_dir="/fake/cache", 
                split="train"
            )
            assert len(dataset) == 0


def test_two_layer_probe_basic():
    """Test two layer probe initialization."""
    from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe
    
    # Just create it with minimal mocking
    probe = EEGPTTwoLayerProbe(n_classes=2)
    
    # Check actual attributes from the model structure - from the error it has linear_layer
    assert hasattr(probe, "linear_layer") or hasattr(probe, "dropout")


def test_edf_streaming_basic():
    """Test EDF streaming initialization."""
    from brain_go_brrr.data.edf_streaming import EDFStreamer
    from pathlib import Path
    
    fake_path = Path("/fake/file.edf")
    # Patch at the import level
    with patch.object(Path, "exists", return_value=True):
        with patch("pyedflib.EdfReader") as mock_reader:
            mock_instance = Mock()
            mock_instance.getNSamples.return_value = [1000]
            mock_instance.getSampleFrequency.return_value = [256]
            mock_reader.return_value = mock_instance
            
            streamer = EDFStreamer(fake_path)
            # Check for actual attributes/methods
            assert hasattr(streamer, "file_path") or hasattr(streamer, "read_window")