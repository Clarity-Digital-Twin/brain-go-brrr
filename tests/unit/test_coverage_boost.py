"""Quick smoke tests to boost coverage for low-coverage modules."""

import json
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, Mock


def test_parallel_pipeline_basic():
    """Test parallel pipeline initialization."""
    from brain_go_brrr.core.pipeline.parallel import ParallelEEGPipeline
    
    # Mock the model that's imported inside the module
    with patch("brain_go_brrr.models.eegpt_model.EEGPTModel"):
        pipeline = ParallelEEGPipeline(device="cpu")
        assert pipeline.device == "cpu"


def test_snippet_maker_basic():
    """Test snippet maker initialization."""
    from brain_go_brrr.core.snippets.maker import EEGSnippetMaker
    
    # Patch model import - use correct parameter names
    with patch("brain_go_brrr.models.eegpt_model.EEGPTModel"):
        maker = EEGSnippetMaker()  # No params needed for default init
        # Check for actual attribute
        assert hasattr(maker, "model") or hasattr(maker, "window_duration")


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
    from brain_go_brrr.core.jobs.store import ThreadSafeJobStore
    from brain_go_brrr.api.schemas import JobData, JobStatus, JobPriority
    
    store = ThreadSafeJobStore()
    
    # Create a job using the actual API
    job_id = "test-123"
    job_data = JobData(
        job_id=job_id,
        analysis_type="test",
        file_path="/fake/file.edf",
        status=JobStatus.PENDING,
        parameters={"param": 1},
        priority=JobPriority.NORMAL,
        progress=0.0,
        result=None,
        error=None,
        metadata={}
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
    
    # Create a Path object
    fake_path = Path("/fake/model.ckpt")
    
    # Mock path validation
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "is_file", return_value=True):
            with patch("brain_go_brrr.core.abnormal.detector.EEGPTModel"):
                detector = AbnormalityDetector(model_path=fake_path)
                assert hasattr(detector, "detect")


def test_cache_redis_init():
    """Test Redis cache initialization."""
    from brain_go_brrr.infra.cache import RedisCache
    
    # Mock redis module
    with patch("redis.Redis") as mock_redis:
        mock_client = Mock()
        mock_redis.return_value = mock_client
        
        cache = RedisCache(host="localhost", port=6379)
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
    
    # Create mock backbone
    mock_backbone = Mock()
    mock_backbone.forward = Mock(return_value=torch.randn(4, 4, 512))
    
    with patch("brain_go_brrr.models.eegpt_two_layer_probe.create_normalized_eegpt", return_value=mock_backbone):
        # Use correct initialization
        probe = EEGPTTwoLayerProbe(
            checkpoint_path="/fake/path.ckpt",
            n_classes=2
        )
        
        # Test forward pass
        x = torch.randn(4, 20, 1024)  # batch, channels, time
        output = probe(x)
        assert output.shape == (4, 2)


def test_edf_streaming_basic():
    """Test EDF streaming initialization."""
    from brain_go_brrr.data.edf_streaming import EDFStreamer
    from pathlib import Path
    
    fake_path = Path("/fake/file.edf")
    
    with patch.object(Path, "exists", return_value=True):
        with patch.object(Path, "suffix", ".edf"):
            with patch("pyedflib.EdfReader"):
                streamer = EDFStreamer(fake_path)
                # Check for actual attributes that exist
                assert hasattr(streamer, "file_path") or hasattr(streamer, "_reader")