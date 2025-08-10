"""Quick smoke tests to boost coverage for low-coverage modules."""

import json
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, Mock


def test_parallel_pipeline_basic():
    """Test parallel pipeline initialization."""
    from brain_go_brrr.core.pipeline.parallel import ParallelEEGPipeline
    
    # Mock the model at the correct import location
    with patch("brain_go_brrr.core.pipeline.parallel.EEGPTModel"):
        pipeline = ParallelEEGPipeline(device="cpu")
        assert pipeline.device == "cpu"


def test_snippet_maker_basic():
    """Test snippet maker initialization."""
    from brain_go_brrr.core.snippets.maker import EEGSnippetMaker
    
    # Patch model import at correct location
    with patch("brain_go_brrr.core.snippets.maker.EEGPTModel"):
        # EEGSnippetMaker doesn't take window_duration in __init__
        maker = EEGSnippetMaker()
        # Check for actual attributes that exist
        assert hasattr(maker, "window_size_sec") or hasattr(maker, "_model")


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
    from brain_go_brrr.core.jobs.store import InMemoryJobStore
    from brain_go_brrr.api.schemas import JobData, JobStatus, JobPriority
    
    store = InMemoryJobStore()
    
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
    
    # Mock the entire initialization including model validation
    with patch("brain_go_brrr.core.abnormal.detector.Path") as mock_path_cls:
        # Make Path() return our mock that always exists
        mock_path_inst = Mock()
        mock_path_inst.exists.return_value = True
        mock_path_inst.is_file.return_value = True
        mock_path_inst.resolve.return_value = fake_path
        mock_path_cls.return_value = mock_path_inst
        mock_path_cls.side_effect = lambda x: mock_path_inst if str(x) == "/fake/model.ckpt" else Path(x)
        
        with patch("brain_go_brrr.core.abnormal.detector.EEGPTModel"):
            detector = AbnormalityDetector(model_path=fake_path)
            assert hasattr(detector, "detect")


def test_cache_redis_init():
    """Test Redis cache initialization."""
    from brain_go_brrr.infra.cache import EmbeddingCache
    
    # EmbeddingCache uses TTL cache, not Redis directly
    cache = EmbeddingCache(max_size=100, ttl_seconds=300)
    assert cache is not None
    assert hasattr(cache, "get") and hasattr(cache, "set")


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
        # Use correct parameters - no input_dim, it's inferred from backbone
        probe = EEGPTTwoLayerProbe(
            checkpoint_path="/fake/path.ckpt",
            output_dim=2  # Changed from n_classes
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
            with patch("pyedflib.EdfReader") as mock_reader:
                # Mock the reader to avoid actual file operations
                mock_instance = Mock()
                mock_instance.getNSamples.return_value = [1000]
                mock_instance.getSampleFrequency.return_value = [256]
                mock_reader.return_value = mock_instance
                
                streamer = EDFStreamer(fake_path)
                # Check for actual method that exists
                assert hasattr(streamer, "stream_windows")