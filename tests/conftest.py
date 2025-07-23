"""Shared test fixtures and configuration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest
from fastapi.testclient import TestClient

# Import benchmark fixtures to make them available
pytest_plugins = ["tests.fixtures.benchmark_data"]


@pytest.fixture(scope="session", autouse=True)
def test_environment_setup():
    """Set up test environment - silence MNE logging and replace Redis."""
    # Silence MNE filter design messages
    import mne

    mne.set_log_level("WARNING")

    # Also set environment variable for any subprocesses
    import os

    os.environ["MNE_LOGGING_LEVEL"] = "WARNING"


@pytest.fixture(scope="session", autouse=True)
def redis_disabled_session():
    """Replace Redis with FakeRedis for all unit tests - session scoped for performance.

    Per senior review: Keep the fake in sys.modules for the whole pytest run
    to avoid restoration issues with background tasks.
    """
    import sys
    import types

    import fakeredis
    import redis as _real_redis

    # Create a fake redis module with only needed attributes
    fake_redis_module = types.ModuleType("redis")
    fake_redis_module.Redis = fakeredis.FakeStrictRedis
    fake_redis_module.StrictRedis = fakeredis.FakeStrictRedis
    # Import real exceptions that fakeredis can actually raise
    fake_redis_module.ConnectionError = _real_redis.ConnectionError
    fake_redis_module.TimeoutError = _real_redis.TimeoutError
    fake_redis_module.RedisError = _real_redis.RedisError

    # Replace in sys.modules - no restoration (keep for whole pytest run)
    sys.modules["redis"] = fake_redis_module

    # Patch the specific import in our pool module
    import brain_go_brrr.infra.redis.pool

    brain_go_brrr.infra.redis.pool.redis = fake_redis_module

    yield

    # No restoration - keep fake module for entire test session


@pytest.fixture(autouse=True)
def fresh_app():
    """Reload api.main module for each test to ensure clean state.

    This fixture addresses the flaky test issue where global state
    (qc_controller) gets mutated by some tests and affects others.
    By reloading the module, we ensure each test starts with a fresh
    instance of the FastAPI app and all its globals.
    """
    # First, clear any existing api.main from sys.modules
    import sys

    if "api.main" in sys.modules:
        del sys.modules["api.main"]
    if "api" in sys.modules:
        del sys.modules["api"]

    # Import fresh

    # The module is now fresh for this test
    yield

    # Cleanup after test
    if "api.main" in sys.modules:
        del sys.modules["api.main"]
    if "api" in sys.modules:
        del sys.modules["api"]


@pytest.fixture
def client():
    """Create a fresh test client with isolated app instance."""
    import brain_go_brrr.api.main as api_main

    return TestClient(api_main.app)


@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sleep_edf_path(project_root) -> Path:
    """Get path to a Sleep-EDF file."""
    edf_path = project_root / "data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf"
    if not edf_path.exists():
        pytest.skip("Sleep-EDF data not available. Run data download scripts first.")
    return edf_path


@pytest.fixture
def sleep_edf_raw_cropped(sleep_edf_path) -> mne.io.Raw:
    """Load Sleep-EDF file cropped to 60 seconds for fast tests."""
    raw = mne.io.read_raw_edf(sleep_edf_path, preload=True)
    raw.crop(tmax=60)  # 1-minute slice for CI speed
    yield raw
    # Cleanup: explicitly delete to free memory
    del raw._data
    del raw


@pytest.fixture
def sleep_edf_raw_full(sleep_edf_path) -> mne.io.Raw:
    """Load full Sleep-EDF file (for slow tests only)."""
    raw = mne.io.read_raw_edf(sleep_edf_path, preload=True)
    yield raw
    # Cleanup: explicitly delete to free memory
    del raw._data
    del raw


@pytest.fixture
def mock_eeg_data():
    """Create mock EEG data for unit tests."""
    # 19 channels, 30 seconds at 256 Hz
    sfreq = 256
    duration = 30
    n_channels = 19
    n_times = int(sfreq * duration)

    ch_names = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",
    ]

    data = np.random.randn(n_channels, n_times) * 20e-6  # ~20 μV

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


@pytest.fixture
def mock_qc_controller():
    """Mock QC controller with proper spec and expected behavior."""
    from brain_go_brrr.core.quality import EEGQualityController

    controller = MagicMock(spec=EEGQualityController)
    controller.eegpt_model = MagicMock()  # Model is loaded
    controller.run_full_qc_pipeline = MagicMock(
        return_value={
            "quality_metrics": {
                "bad_channels": ["T3", "O2"],
                "bad_channel_ratio": 0.21,
                "abnormality_score": 0.82,
                "quality_grade": "POOR",
                "total_channels": 19,
                "artifact_ratio": 0.15,
            },
            "processing_info": {"confidence": 0.85},
            "processing_time": 1.5,
        }
    )
    return controller


@pytest.fixture(scope="session")
def tiny_edf(tmp_path_factory):
    """Create a tiny, valid EDF file using pyEDFlib."""
    import numpy as np
    from pyedflib import EdfWriter

    # Create a temporary path for the EDF file
    path = tmp_path_factory.mktemp("edf") / "tiny.edf"

    # Create the EDF writer with 1 channel
    writer = EdfWriter(str(path), n_channels=1)

    # Set signal header for one EEG channel
    writer.setSignalHeader(
        0,
        {
            "label": "EEG Fpz-Cz",
            "dimension": "uV",
            "sample_frequency": 256,
            "physical_max": 250,
            "physical_min": -250,
            "digital_max": 2047,
            "digital_min": -2048,
            "prefilter": "HP:0.1Hz LP:75Hz",
            "transducer": "AgAgCl electrode",
        },
    )

    # Write 1 second of zero data (256 samples at 256 Hz)
    data = np.zeros(256, dtype=np.int32)
    writer.writeDigitalSamples(data)

    # Close the writer to finalize the file
    writer.close()

    # Return the file contents as bytes
    return path.read_bytes()


@pytest.fixture
def valid_edf_content(tiny_edf):
    """Alias for tiny_edf for backward compatibility."""
    return tiny_edf


@pytest.fixture
def valid_edf_file(valid_edf_content):
    """Create a temporary valid EDF file."""
    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as f:
        f.write(valid_edf_content)
        f.flush()
        yield Path(f.name)
    # Cleanup
    if Path(f.name).exists():
        Path(f.name).unlink()


@pytest.fixture
def patched_qc_endpoint(mock_qc_controller):
    """Provide a context manager for patching QC endpoint dependencies."""

    def _patch():
        return [
            patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller),
        ]

    return _patch


@pytest.fixture
def mock_abnormality_detector():
    """Mock abnormality detector with proper spec."""
    from brain_go_brrr.core.abnormal.detector import AbnormalityDetector

    detector = MagicMock(spec=AbnormalityDetector)
    detector.detect_abnormality = MagicMock(
        return_value={
            "abnormal": False,
            "confidence": 0.85,
            "probabilities": {"normal": 0.85, "abnormal": 0.15},
        }
    )
    return detector
