"""Shared test fixtures and configuration."""

from __future__ import annotations

import os

# MUST disable multiprocessing BEFORE any imports to prevent hangs
os.environ["MNE_USE_NUMBA"] = "false"
os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "sequential"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

import random
import socket
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Type checking imports only - don't trigger actual imports
if TYPE_CHECKING:
    import mne

# Import benchmark fixtures to make them available
pytest_plugins = ["tests.fixtures.benchmark_data", "tests.fixtures.cache_fixtures"]

# Set deterministic random seeds for reproducible tests
random.seed(1337)
np.random.seed(1337)


def can_connect_to_redis(host="localhost", port=6379, timeout=0.5):
    """Check if Redis is available for integration tests."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: needs large models or datasets")
    config.addinivalue_line("markers", "slow: test takes > 5 seconds")
    config.addinivalue_line("markers", "external: requires external services or data")
    config.addinivalue_line("markers", "redis: requires Redis server")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless --run-integration is passed."""
    if config.getoption("--run-integration", default=False):
        return

    skip_integration = pytest.mark.skip(reason="integration test (run with --run-integration)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that require models/data",
    )


@pytest.fixture(autouse=True)
def force_seq_joblib(monkeypatch):
    """Force sequential execution for joblib to prevent hangs."""
    # Make anything that reads this var stay single-threaded
    monkeypatch.setenv("JOBLIB_START_METHOD", "threading")
    monkeypatch.setenv("JOBLIB_N_JOBS", "1")
    monkeypatch.setenv("AUTOREJECT_N_JOBS", "1")
    monkeypatch.setenv("SKLEARN_N_JOBS", "1")
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")


@pytest.fixture(scope="session")
def mne_mod():
    """Import MNE and silence its logging - safe runtime import."""
    import mne

    os.environ["MNE_LOGGING_LEVEL"] = "WARNING"
    mne.set_log_level("WARNING")
    return mne


@pytest.fixture(scope="session", autouse=True)
def test_environment_setup():
    """Set up test environment - configure logging levels."""
    # Set environment variable for any subprocesses
    os.environ["MNE_LOGGING_LEVEL"] = "WARNING"


# DummyCache and cache fixtures are now in tests.fixtures.cache_fixtures


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


# client_with_cache fixture is now in tests.fixtures.cache_fixtures


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
def sleep_edf_raw_cropped(sleep_edf_path, mne_mod) -> mne.io.Raw:
    """Load Sleep-EDF file cropped to 60 seconds for fast tests."""
    mne = mne_mod
    raw = mne.io.read_raw_edf(sleep_edf_path, preload=True)
    raw.crop(tmax=60)  # 1-minute slice for CI speed
    yield raw
    # Cleanup: explicitly delete to free memory
    del raw._data
    del raw


@pytest.fixture
def sleep_edf_raw_full(sleep_edf_path, mne_mod) -> mne.io.Raw:
    """Load full Sleep-EDF file (for slow tests only)."""
    mne = mne_mod
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

    # Write 30 seconds of zero data (7680 samples at 256 Hz)
    # This gives us 7 windows of 4 seconds each for EEGPT
    data = np.zeros(30 * 256, dtype=np.int32)
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


@pytest.fixture
def channel_shuffled_raw(mock_eeg_data, mne_mod):
    """Create mock EEG data with shuffled channel order.

    This fixture creates EEG data with channels in a randomized order
    to test robustness of algorithms to different channel arrangements.
    """
    mne = mne_mod

    # Set seed for reproducible shuffling
    np.random.seed(42)

    # Get the original data
    raw = mock_eeg_data.copy()

    # Get channel names and indices
    ch_names = raw.ch_names.copy()
    n_channels = len(ch_names)

    # Create a shuffled order
    shuffled_indices = np.random.permutation(n_channels)
    shuffled_ch_names = [ch_names[i] for i in shuffled_indices]

    # Get the data and shuffle it
    data = raw.get_data()
    shuffled_data = data[shuffled_indices, :]

    # Create new info with shuffled channel order
    info = mne.create_info(ch_names=shuffled_ch_names, sfreq=raw.info["sfreq"], ch_types="eeg")

    # Create new Raw object with shuffled data
    shuffled_raw = mne.io.RawArray(shuffled_data, info)

    return shuffled_raw


@pytest.fixture(autouse=False)  # DISABLED - causes import hangs
def mock_eegpt_model(monkeypatch):
    """Auto-mock EEGPT model loading for all unit tests."""
    if os.environ.get("EEGPT_MODEL_PATH"):
        # If model path is set, don't mock - allow real loading
        return

    # Use centralized mocks
    from tests._mocks import mock_eegpt_model_loading

    mock_eegpt_model_loading(monkeypatch)


@pytest.fixture(autouse=True, scope="session")
def _isolate_autoreject_cache(tmp_path_factory):
    """Isolate AutoReject cache to temp directory for tests."""
    tmp = tmp_path_factory.mktemp("ar_cache")
    import os

    os.environ["BGB_AR_CACHE_DIR"] = str(tmp)
    yield
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
