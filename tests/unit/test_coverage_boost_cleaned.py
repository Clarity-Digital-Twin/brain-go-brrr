"""Clean coverage boost tests targeting actual existing modules."""

from pathlib import Path
from unittest.mock import patch

import numpy as np


def test_api_health_check():
    """Test health check endpoint."""
    import asyncio

    from brain_go_brrr.api.routers.health import health_check

    # Run async function
    result = asyncio.run(health_check())

    assert result["status"] == "healthy"
    assert "timestamp" in result
    assert "service" in result
    assert result["service"] == "brain-go-brrr-api"


def test_api_readiness_check():
    """Test readiness check endpoint."""
    import asyncio

    from brain_go_brrr.api.routers.health import readiness_check

    # Run async function
    result = asyncio.run(readiness_check())

    assert result["status"] == "ready"
    assert "timestamp" in result


def test_job_status_enum():
    """Test job status enum values."""
    from brain_go_brrr.api.schemas import JobStatus

    assert JobStatus.PENDING.value == "pending"
    assert JobStatus.PROCESSING.value == "processing"
    assert JobStatus.COMPLETED.value == "completed"
    assert JobStatus.FAILED.value == "failed"


def test_job_priority_enum():
    """Test job priority enum values."""
    from brain_go_brrr.api.schemas import JobPriority

    assert JobPriority.LOW.value == "low"
    assert JobPriority.NORMAL.value == "normal"
    assert JobPriority.HIGH.value == "high"


def test_analysis_type_enum():
    """Test analysis type enum values - skip if not implemented."""
    pytest.skip("AnalysisType not yet implemented in api.schemas")


def test_eegpt_config_properties():
    """Test EEGPT config computed properties."""
    from brain_go_brrr.models.eegpt_model import EEGPTConfig

    config = EEGPTConfig(window_duration=4.0, sampling_rate=256)

    # Test computed properties
    assert config.window_samples == 1024  # 4 * 256
    assert config.n_patches_per_window == 16  # 1024 / 64


def test_triage_level_enum():
    """Test triage level enum values."""
    from brain_go_brrr.core.abnormal.detector import TriageLevel

    assert TriageLevel.NORMAL.value == "NORMAL"
    assert TriageLevel.ROUTINE.value == "ROUTINE"
    assert TriageLevel.EXPEDITE.value == "EXPEDITE"
    assert TriageLevel.URGENT.value == "URGENT"


def test_aggregation_method_enum():
    """Test aggregation method enum values."""
    from brain_go_brrr.core.abnormal.detector import AggregationMethod

    assert AggregationMethod.WEIGHTED_AVERAGE.value == "weighted_average"
    assert AggregationMethod.VOTING.value == "voting"
    assert AggregationMethod.ATTENTION.value == "attention"


def test_window_result_dataclass():
    """Test WindowResult dataclass."""
    from brain_go_brrr.core.abnormal.detector import WindowResult

    result = WindowResult(
        index=0,
        start_time=0.0,
        end_time=4.0,
        abnormality_score=0.75,
        quality_score=0.9
    )

    assert result.index == 0
    assert result.start_time == 0.0
    assert result.end_time == 4.0
    assert result.abnormality_score == 0.75
    assert result.quality_score == 0.9


def test_validation_result_dataclass():
    """Test ValidationResult dataclass."""
    from brain_go_brrr.core.edf_validator import ValidationResult

    result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=["Sample warning"],
        metadata={"channels": 19}
    )

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 1
    assert result.metadata["channels"] == 19


def test_snippet_maker_init():
    """Test snippet maker initialization."""
    from brain_go_brrr.core.snippets.maker import EEGSnippetMaker

    maker = EEGSnippetMaker(snippet_length=4.0, overlap=0.5)

    assert maker.snippet_length == 4.0
    assert maker.overlap == 0.5


def test_edf_validator_init():
    """Test EDF validator initialization."""
    from brain_go_brrr.core.edf_validator import EDFValidator

    validator = EDFValidator(
        min_duration_seconds=30.0,
        min_channels=10,
        max_amplitude_v=1e-3
    )

    assert validator.min_duration_seconds == 30.0
    assert validator.min_channels == 10
    assert validator.max_amplitude_v == 1e-3


def test_model_config_defaults():
    """Test ModelConfig default values."""
    from brain_go_brrr.core.config import ModelConfig

    config = ModelConfig()

    # Check defaults
    assert config.device in ["cpu", "cuda", "auto"]
    assert config.batch_size > 0
    assert config.sampling_rate == 256
    assert config.window_duration == 4.0


def test_abnormality_config_defaults():
    """Test AbnormalityConfig default values."""
    from brain_go_brrr.core.abnormality_config import AbnormalityConfig

    config = AbnormalityConfig()

    # Check that config has expected attributes
    assert hasattr(config, "classification")
    assert hasattr(config, "quality")
    assert hasattr(config, "processing")  # Fixed: processing not preprocessing


def test_sleep_stage_mapping():
    """Test sleep stage label mapping."""
    from brain_go_brrr.services.yasa_adapter import YASASleepStager

    stager = YASASleepStager()

    # Test stage mapping
    assert stager.stage_labels[0] == "W"
    assert stager.stage_labels[1] == "N1"
    assert stager.stage_labels[2] == "N2"
    assert stager.stage_labels[3] == "N3"
    assert stager.stage_labels[4] == "REM"


def test_feature_extractor_device():
    """Test feature extractor device handling."""
    from brain_go_brrr.core.features.extractor import EEGPTFeatureExtractor

    # Test with CPU device
    with patch("brain_go_brrr.core.features.extractor.EEGPTModel") as mock_model:
        extractor = EEGPTFeatureExtractor(device="cpu")
        assert extractor.device == "cpu"


def test_chunked_autoreject_init():
    """Test chunked autoreject processor initialization."""
    from brain_go_brrr.preprocessing.chunked_autoreject import ChunkedAutoRejectProcessor

    processor = ChunkedAutoRejectProcessor(
        chunk_size=10,
        n_interpolate=[1, 2],
        consensus=[0.1, 0.2],
        cv=3
    )

    assert processor.chunk_size == 10
    assert processor.n_interpolate == [1, 2]
    assert processor.consensus == [0.1, 0.2]
    assert processor.cv == 3


def test_channel_mapping():
    """Test EEG channel mapping and validation."""
    from brain_go_brrr.core.channels import ChannelMapper

    mapper = ChannelMapper()

    # Test old to new channel mapping
    old_channels = ["T3", "T4", "T5", "T6"]
    new_channels = mapper.map_old_to_new(old_channels)

    assert "T7" in new_channels  # T3 -> T7
    assert "T8" in new_channels  # T4 -> T8
    assert "P7" in new_channels  # T5 -> P7
    assert "P8" in new_channels  # T6 -> P8


def test_window_extractor():
    """Test EEG window extraction."""
    from brain_go_brrr.core.window_extractor import WindowExtractor

    # Create fake EEG data
    sfreq = 256
    duration = 20
    n_channels = 19
    data = np.random.randn(n_channels, sfreq * duration) * 1e-6

    extractor = WindowExtractor(
        window_duration=4.0,
        window_stride=2.0,
        sfreq=sfreq
    )

    windows = extractor.extract_windows(data)

    # Should have (20-4)/2 + 1 = 9 windows
    assert len(windows) == 9
    assert windows[0].shape == (n_channels, 4 * sfreq)


def test_api_cache_init():
    """Test API cache initialization."""
    from brain_go_brrr.api.cache import CacheManager

    with patch("brain_go_brrr.api.cache.redis.Redis"):
        cache = CacheManager(
            host="localhost",
            port=6379,
            ttl=300
        )

        assert cache.host == "localhost"
        assert cache.port == 6379
        assert cache.ttl == 300


def test_infra_cache_init():
    """Test infrastructure cache initialization."""
    from brain_go_brrr.infra.cache import InfraCache

    cache = InfraCache(
        cache_dir=Path("/tmp/cache"),
        max_size_gb=10
    )

    assert cache.cache_dir == Path("/tmp/cache")
    assert cache.max_size_gb == 10


def test_rotary_embedding():
    """Test rotary position embeddings."""
    from brain_go_brrr.models.eegpt_architecture import RoPE

    rope = RoPE(dim=64, theta=10000.0)

    assert rope.dim == 64
    assert rope.theta == 10000.0

    # Test frequency preparation
    freqs = rope.prepare_freqs((1, 16), device="cpu")
    assert freqs.shape[0] == 16  # Number of patches


def test_eeg_transformer_init():
    """Test EEG Transformer initialization."""
    from brain_go_brrr.models.eegpt_architecture import EEGTransformer

    model = EEGTransformer(
        n_channels=["C3", "C4", "Cz"],
        patch_size=64,
        embed_dim=256,
        depth=4,
        num_heads=4
    )

    assert len(model.n_channels) == 3
    assert model.patch_size == 64
    assert model.embed_dim == 256
    assert len(model.blocks) == 4


def test_abnormality_result_to_dict():
    """Test AbnormalityResult conversion to dict."""
    from brain_go_brrr.core.abnormal.detector import AbnormalityResult, TriageLevel, WindowResult

    window_results = [
        WindowResult(
            index=0,
            start_time=0.0,
            end_time=4.0,
            abnormality_score=0.75,
            quality_score=0.9
        )
    ]

    result = AbnormalityResult(
        abnormality_score=0.8,
        classification="abnormal",
        confidence=0.9,
        triage_flag=TriageLevel.EXPEDITE,
        window_scores=window_results,
        quality_metrics={"quality_grade": "GOOD"},
        processing_time=1.5,
        model_version="v1.0"
    )

    result_dict = result.to_dict()

    assert result_dict["abnormality_score"] == 0.8
    assert result_dict["classification"] == "abnormal"
    assert result_dict["triage_flag"] == "EXPEDITE"
    assert len(result_dict["window_scores"]) == 1
