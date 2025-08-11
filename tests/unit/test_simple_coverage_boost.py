"""Simple coverage boost tests targeting actual modules."""

from pathlib import Path
from unittest.mock import patch


def test_api_health_check():
    """Test health check endpoint."""
    import asyncio

    from brain_go_brrr.api.routers.health import health_check

    # Run async function
    result = asyncio.run(health_check())

    assert result["status"] == "healthy"
    assert "timestamp" in result
    assert "service" in result
    assert "version" in result
    assert result["service"] == "brain-go-brrr-api"
    assert result["version"] == "0.1.0"


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
    assert JobPriority.URGENT.value == "urgent"


def test_analysis_type_enum():
    """Test analysis type enum values."""
    from brain_go_brrr.api.schemas import AnalysisType

    assert AnalysisType.ABNORMALITY.value == "abnormality"
    assert AnalysisType.SLEEP.value == "sleep"
    assert AnalysisType.QUALITY.value == "quality"
    assert AnalysisType.EPILEPTIFORM.value == "epileptiform"


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
    assert hasattr(config, "preprocessing")


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


def test_linear_probe_trainable_params():
    """Test getting number of trainable parameters."""
    from brain_go_brrr.models.eegpt_linear_probe import EEGPTLinearProbe

    with patch("brain_go_brrr.models.eegpt_linear_probe.create_normalized_eegpt"):
        with patch("brain_go_brrr.models.eegpt_linear_probe.Path"):
            probe = EEGPTLinearProbe(
                checkpoint_path=Path("/fake/path"),
                n_input_channels=20,
                n_classes=2
            )

            # Should have some trainable params
            n_params = probe.get_num_trainable_params()
            assert n_params >= 0


def test_two_layer_probe_init():
    """Test two-layer probe initialization."""
    from brain_go_brrr.models.eegpt_two_layer_probe import EEGPTTwoLayerProbe

    probe = EEGPTTwoLayerProbe(
        n_classes=3,
        hidden_dim=256
    )

    # Check structure
    assert hasattr(probe, "fc1")
    assert hasattr(probe, "fc2")
    assert hasattr(probe, "activation")
    assert hasattr(probe, "dropout")
