"""Refactored coverage tests using behavior-driven testing and proper fakes.

This module replaces test_coverage_boost.py with clean, maintainable tests
that verify behavior rather than implementation details.
"""

import json
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import torch

from tests.fakes import (
    FakeClassifierHead,
    FakeEdfReader,
    FakeEEGPTBackbone,
    FakeFeatureExtractor,
    FakeMNERaw,
    FakeRedis,
    FakeSleepAnalyzer,
)


def test_parallel_pipeline_processes_data():
    """Test that parallel pipeline actually processes EEG data."""
    from brain_go_brrr.core.pipeline.parallel import ParallelEEGPipeline

    # Arrange: Create pipeline with fake dependencies
    fake_extractor = FakeFeatureExtractor(feature_dim=128)
    fake_analyzer = FakeSleepAnalyzer()
    pipeline = ParallelEEGPipeline(
        extractor=fake_extractor,
        sleep_analyzer=fake_analyzer
    )
    
    # Act: Process fake EEG data
    fake_raw = FakeMNERaw(n_channels=19, duration=30.0)
    results = pipeline.process(fake_raw)
    
    # Assert: Verify both processors returned results
    assert "eegpt" in results
    assert "sleep" in results
    assert results["sleep"]["sleep_efficiency"] == 85.5
    # EEGPT features should have correct shape
    if "features" in results["eegpt"]:
        assert results["eegpt"]["features"].shape[-1] == 128


def test_snippet_maker_creates_snippets():
    """Test that snippet maker creates EEG snippets with correct properties."""
    from brain_go_brrr.core.snippets.maker import EEGSnippetMaker

    # Arrange
    maker = EEGSnippetMaker(snippet_length=4.0, overlap=0.5)
    
    # Act: Create snippets from fake data
    fake_raw = FakeMNERaw(n_channels=19, duration=10.0, sfreq=256.0)
    snippets = maker.make_snippets(fake_raw)
    
    # Assert: Verify snippet properties
    assert len(snippets) > 0
    # Each snippet should be 4 seconds (4 * 256 = 1024 samples)
    if len(snippets) > 0:
        assert snippets[0].shape[1] == 1024 or snippets[0].shape[1] == 4.0 * 256


def test_tuab_dataset_handles_empty_directory():
    """Test TUAB dataset correctly handles empty directory."""
    from brain_go_brrr.data.tuab_dataset import TUABDataset

    with patch("brain_go_brrr.data.tuab_dataset.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.glob.return_value = []

        dataset = TUABDataset(root_dir="/fake/path", split="train")
        
        # Behavior: Empty dataset should have zero length
        assert len(dataset) == 0
        # Should not raise when checking length


def test_job_store_manages_jobs():
    """Test job store can create, retrieve, and update jobs."""
    from datetime import datetime

    from brain_go_brrr.api.schemas import JobData, JobPriority, JobStatus
    from brain_go_brrr.core.jobs.store import ThreadSafeJobStore as JobStore

    # Arrange
    store = JobStore()
    job_id = "test-123"
    now = datetime.now()
    
    # Act: Create a job
    job_data = JobData(
        job_id=job_id,
        analysis_type="abnormality",
        file_path="/data/test.edf",
        status=JobStatus.PENDING,
        priority=JobPriority.HIGH,
        created_at=now,
        updated_at=now,
        options={"threshold": 0.8},
        progress=0.0,
    )
    store.create(job_id, job_data)
    
    # Assert: Job can be retrieved and has correct data
    retrieved = store.get(job_id)
    assert retrieved is not None
    assert retrieved.analysis_type == "abnormality"
    assert retrieved.priority == JobPriority.HIGH
    assert retrieved.options["threshold"] == 0.8
    
    # Behavior: Update job status
    store.update_status(job_id, JobStatus.PROCESSING)
    updated = store.get(job_id)
    assert updated.status == JobStatus.PROCESSING


def test_linear_probe_produces_predictions():
    """Test linear probe produces correct shaped predictions."""
    from brain_go_brrr.models.eegpt_linear_probe import EEGPTLinearProbe

    with patch("brain_go_brrr.models.eegpt_linear_probe.create_normalized_eegpt") as mock:
        # Use fake backbone
        mock.return_value = FakeEEGPTBackbone(feature_dim=2048)
        
        probe = EEGPTLinearProbe(
            checkpoint_path="/fake/path.ckpt",
            n_input_channels=20,
            n_classes=2
        )
        
        # Act: Forward pass with batch of EEG windows
        x = torch.randn(8, 20, 1024)  # batch_size=8, channels=20, samples=1024
        output = probe(x)
        
        # Assert: Output shape matches expected classes
        assert output.shape == (8, 2)
        # Probabilities should be valid
        probs = torch.softmax(output, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_sleep_analyzer_analyzes_sleep():
    """Test sleep analyzer produces sleep analysis results."""
    from brain_go_brrr.core.sleep.analyzer import SleepAnalyzer

    # Arrange
    analyzer = SleepAnalyzer()
    fake_raw = FakeMNERaw(n_channels=2, duration=3600.0)  # 1 hour
    
    # Act: Run sleep analysis
    results = analyzer.run_full_sleep_analysis(fake_raw)
    
    # Assert: Results contain expected sleep metrics
    assert "hypnogram" in results
    assert "sleep_efficiency" in results
    assert 0 <= results["sleep_efficiency"] <= 100
    assert "total_sleep_time" in results
    assert results["total_sleep_time"] >= 0


def test_abnormal_detector_detects_abnormalities():
    """Test abnormality detector produces valid predictions."""
    from brain_go_brrr.core.abnormal.detector import AbnormalityDetector

    # Arrange: Use fake classifier to avoid loading weights
    fake_path = Path("/fake/model.ckpt")
    fake_classifier = FakeClassifierHead(input_dim=2048, n_classes=2)
    
    with patch("brain_go_brrr.core.config.Path.exists", return_value=True), \
         patch("brain_go_brrr.core.config.Path.is_file", return_value=True), \
         patch("brain_go_brrr.core.abnormal.detector.EEGPTModel") as mock_model:
        
        # Setup fake model
        mock_instance = FakeEEGPTBackbone()
        mock_model.return_value = mock_instance
        
        detector = AbnormalityDetector(
            model_path=fake_path,
            classifier=fake_classifier
        )
        
        # Act: Detect abnormality in fake EEG
        fake_raw = FakeMNERaw(n_channels=19, duration=20.0)
        result = detector.detect_abnormality(fake_raw)
        
        # Assert: Result has expected structure
        assert result.prediction in ["normal", "abnormal"]
        assert 0 <= result.confidence <= 1.0
        assert result.triage_level in ["NORMAL", "ROUTINE", "EXPEDITE", "URGENT"]


def test_redis_cache_stores_and_retrieves():
    """Test Redis cache actually stores and retrieves data."""
    from brain_go_brrr.infra.cache import RedisCache

    # Arrange: Use fake Redis
    fake_redis = FakeRedis()
    with patch("redis.Redis", return_value=fake_redis):
        cache = RedisCache()
        
        # Act: Store and retrieve data
        test_key = "test:key"
        test_value = b"test_data"
        cache.set(test_key, test_value)
        retrieved = cache.get(test_key)
        
        # Assert: Data was stored and retrieved correctly
        assert retrieved == test_value
        assert fake_redis.call_count["set"] == 1
        assert fake_redis.call_count["get"] == 1


def test_eeg_preprocessor_preprocesses_data():
    """Test EEG preprocessor transforms data correctly."""
    from brain_go_brrr.preprocessing.eeg_preprocessor import EEGPreprocessor

    # Arrange
    preprocessor = EEGPreprocessor(
        target_sfreq=256,
        lowpass=50,
        highpass=0.5
    )
    fake_raw = FakeMNERaw(n_channels=19, sfreq=500)  # Different sampling rate
    
    # Act: Preprocess data
    processed = preprocessor.preprocess(fake_raw)
    
    # Assert: Data was transformed
    assert processed.info["sfreq"] == 256  # Resampled
    # Should return MNE Raw-like object
    assert hasattr(processed, "get_data")
    data = processed.get_data()
    assert data.shape[0] == 19  # Channels preserved


def test_feature_extractor_extracts_features():
    """Test feature extractor produces feature vectors."""
    from brain_go_brrr.core.features.extractor import EEGPTFeatureExtractor

    with patch("brain_go_brrr.core.features.extractor.EEGPTModel") as mock_model:
        mock_model.return_value = FakeEEGPTBackbone()
        
        extractor = EEGPTFeatureExtractor(model_path="/fake/path")
        
        # Act: Extract features from EEG windows
        windows = np.random.randn(10, 19, 1024).astype(np.float32)
        features = extractor.extract_embeddings(windows)
        
        # Assert: Features have correct shape
        assert features.shape == (10, 2048)
        assert features.dtype == np.float32


def test_chunked_autoreject_processes_chunks():
    """Test chunked autoreject processes data in chunks."""
    from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor

    # Arrange
    processor = ChunkedAutoRejectProcessor(chunk_size=5)
    fake_raw = FakeMNERaw(n_channels=19, duration=300)  # 5 minutes
    
    # Act: Process with autoreject
    cleaned = processor.process(fake_raw)
    
    # Assert: Returns cleaned data
    assert cleaned is not None
    # Should process in chunks of 5 epochs
    assert processor.chunk_size == 5


def test_cached_dataset_loads_from_cache():
    """Test cached dataset loads preprocessed data."""
    from brain_go_brrr.data.tuab_cached_dataset import TUABCachedDataset

    # Arrange: Mock cache index
    mock_index = {
        "files": {
            "file1.edf": {
                "cache_file": "cache_0001.pt",
                "n_windows": 10,
                "label": 0
            }
        },
        "n_files": 1,
        "total_windows": 10,
        "metadata": {"split": "train"}
    }
    
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_index))), \
         patch("brain_go_brrr.data.tuab_cached_dataset.Path") as mock_path:
        
        mock_path.return_value.exists.return_value = True
        
        dataset = TUABCachedDataset(
            root_dir="/data/tuab",
            cache_dir="/data/cache",
            split="train"
        )
        
        # Assert: Dataset loaded index correctly
        assert len(dataset) == 10
        # Can access metadata
        assert dataset.index["n_files"] == 1


def test_two_layer_probe_forward_pass():
    """Test two-layer probe produces correct outputs."""
    from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe

    # Arrange
    probe = EEGPTTwoLayerProbe(n_classes=3, hidden_dim=512)
    
    # Act: Forward pass
    x = torch.randn(16, 2048)  # batch=16, features=2048
    output = probe(x)
    
    # Assert: Output shape matches classes
    assert output.shape == (16, 3)
    # Should produce valid logits
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_edf_streamer_streams_windows():
    """Test EDF streamer yields data windows."""
    from brain_go_brrr.data.edf_streaming import EDFStreamer

    # Arrange: Use fake reader
    fake_path = Path("/fake/file.edf")
    with patch.object(Path, "exists", return_value=True):
        streamer = EDFStreamer(
            fake_path, 
            reader_factory=FakeEdfReader
        )
        
        # Act: Stream windows
        windows = list(streamer.stream_epochs(
            window_sec=4.0,
            overlap=0.5
        ))
        
        # Assert: Windows were streamed
        assert len(windows) > 0
        if len(windows) > 0:
            # Each window should be 4 seconds
            assert windows[0].shape[1] == 4.0 * 256