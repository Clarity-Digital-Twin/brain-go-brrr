"""Quick smoke tests to boost coverage for low-coverage modules."""

import json
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open


def test_parallel_pipeline_basic():
    """Test parallel pipeline initialization."""
    from brain_go_brrr.core.pipeline.parallel import ParallelEEGPipeline
    
    # Patch at the correct import location
    with patch("brain_go_brrr.core.pipeline.parallel.EEGPTModel"):
        pipeline = ParallelEEGPipeline(max_workers=2)
        assert pipeline.max_workers == 2


def test_snippet_maker_basic():
    """Test snippet maker initialization."""
    from brain_go_brrr.core.snippets.maker import EEGSnippetMaker
    
    # Patch model import
    with patch("brain_go_brrr.models.eegpt_model.EEGPTModel"):
        maker = EEGSnippetMaker(window_duration=4.0, overlap=0.5)
        assert maker.window_duration == 4.0


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
    from brain_go_brrr.core.jobs.store import InMemoryJobStore as JobStore
    
    store = JobStore()
    job_id = store.create_job("test", {"param": 1})
    assert job_id is not None
    
    job = store.get_job(job_id) 
    assert job.job_type == "test"


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
    
    with patch("brain_go_brrr.core.abnormal.detector.EEGPTModel"):
        # Add required model_path parameter
        detector = AbnormalityDetector(model_path="/fake/model.ckpt")
        assert hasattr(detector, "detect")


def test_cache_manager_init(tmp_path):
    """Test cache manager with temp directory."""
    from brain_go_brrr.infra.cache import EmbeddingCache
    
    cache = EmbeddingCache(cache_dir=tmp_path)
    # Test basic operation
    cache.set("test_key", np.array([1, 2, 3]))
    assert cache.get("test_key") is not None


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
    
    probe = EEGPTTwoLayerProbe(
        input_dim=512,
        hidden_dim=256,
        n_classes=2
    )
    
    # Test forward pass
    x = torch.randn(4, 512)
    output = probe(x)
    assert output.shape == (4, 2)


def test_edf_streaming_basic():
    """Test EDF streaming initialization."""
    from brain_go_brrr.data.edf_streaming import EDFStreamer
    
    with patch("brain_go_brrr.data.edf_streaming.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.suffix = ".edf"
        with patch("pyedflib.EdfReader"):
            streamer = EDFStreamer("/fake/file.edf")
            assert hasattr(streamer, "stream_windows")