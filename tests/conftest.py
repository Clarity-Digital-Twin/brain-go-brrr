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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Seed randomness once for reproducible tests
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

# Only import and seed torch if available
try:
    import torch

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
except Exception:
    # Broader catch to handle any torch import issues
    torch = None


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and options.

    Handles:
    1. Skip network tests when BGB_ALLOW_NET is not set
    2. Deselect integration tests unless --run-integration is passed
    3. Skip data tests unless --run-data is passed AND BGB_DATA_ROOT exists
    4. Skip GPU tests when CUDA is not available
    """
    from pathlib import Path

    import torch

    # Handle network tests
    allow_network = os.environ.get("BGB_ALLOW_NET", "0") == "1"
    if not allow_network:
        skip_network = pytest.mark.skip(reason="Network disabled (set BGB_ALLOW_NET=1)")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)

    # Handle integration tests
    if not config.getoption("--run-integration", default=False):
        # Deselect instead of skip for cleaner output
        drop = [it for it in items if "integration" in it.keywords]
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = [it for it in items if it not in drop]

    # Handle data-backed tests
    run_data = config.getoption("--run-data", default=False)
    data_root = os.environ.get("BGB_DATA_ROOT", "")
    has_data = bool(data_root and Path(data_root).exists())

    if not (run_data and has_data):
        reason = "skipping data-backed tests (set BGB_DATA_ROOT and pass --run-data)"
        skip_data = pytest.mark.skip(reason=reason)
        for item in items:
            if "data" in item.keywords:
                item.add_marker(skip_data)

    # Handle GPU tests
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="no CUDA available in CI")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# Type checking imports only - don't trigger actual imports

# Import benchmark fixtures to make them available
# benchmark_data causes pytest to hang during collection - disabled
pytest_plugins = [
    # "tests.fixtures.benchmark_data",  # DISABLED - causes hang
    "tests.fixtures.cache_fixtures",
    "tests.fixtures.synthetic_data",
    "tests.fixtures.deterministic_fixtures",  # Add centralized fixtures with deterministic seed
]

# Seeds already set above with SEED=1337


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


@pytest.fixture
def redis_client():
    """Provide Redis client - real if available, fake otherwise."""
    if can_connect_to_redis():
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0)
        yield client
        client.flushdb()  # Clean up after test
    else:
        import fakeredis

        client = fakeredis.FakeRedis()
        yield client
        client.flushdb()


@pytest.fixture
def fake_redis():
    """Always provide fake Redis for unit tests."""
    import fakeredis

    client = fakeredis.FakeRedis()
    yield client
    client.flushdb()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: needs large models or datasets")
    config.addinivalue_line("markers", "slow: test takes > 5 seconds")
    config.addinivalue_line("markers", "external: requires external services or data")
    config.addinivalue_line("markers", "redis: requires Redis server")
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "benchmark: benchmark test")
    config.addinivalue_line("markers", "network: requires network access")


def pytest_sessionstart(session):
    """Session start hook - seeds already set at module level above."""
    # Seeds are already set at module level (lines 28-41)
    # No need to duplicate here
    pass


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests that require models/data",
    )
    parser.addoption(
        "--run-data",
        action="store_true",
        default=False,
        help="run tests that require real datasets (TUAB/TUEV/Sleep-EDF)",
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


@pytest.fixture
def fresh_app():
    """Reload api.main module for each test to ensure clean state.

    This fixture addresses the flaky test issue where global state
    (qc_controller) gets mutated by some tests and affects others.
    By reloading the module, we ensure each test starts with a fresh
    instance of the FastAPI app and all its globals.

    NOTE: This is now opt-in - only tests that mutate global app state
    should use this fixture.
    """
    # First, clear any existing api.main from sys.modules
    import sys

    # Delete the correct module names
    if "brain_go_brrr.api.main" in sys.modules:
        del sys.modules["brain_go_brrr.api.main"]
    if "brain_go_brrr.api" in sys.modules:
        del sys.modules["brain_go_brrr.api"]

    # Import fresh

    # The module is now fresh for this test
    yield

    # Cleanup after test
    # Delete the correct module names
    if "brain_go_brrr.api.main" in sys.modules:
        del sys.modules["brain_go_brrr.api.main"]
    if "brain_go_brrr.api" in sys.modules:
        del sys.modules["brain_go_brrr.api"]


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
    """Get path to a Sleep-EDF PSG file from config.

    Uses DataConfig to resolve paths deterministically.
    """
    from brain_go_brrr.application.config import DataConfig

    config = DataConfig(data_path=project_root / "data")
    path = config.get_sleep_edf_psg_file()
    if not path:
        pytest.skip(
            "Sleep-EDF data not available. Set BGB_SLEEP_EDF_DIR or BGB_DATA_ROOT and pass --run-data."
        )
    return path


@pytest.fixture
def sleep_edf_dir(project_root) -> Path:
    """Get Sleep-EDF directory from config.

    Uses DataConfig to resolve paths deterministically.
    """
    from brain_go_brrr.application.config import DataConfig

    config = DataConfig(data_path=project_root / "data")
    dir_path = config.sleep_edf_cassette_dir
    if not dir_path.exists():
        pytest.skip(
            "Sleep-EDF directory not available. Set BGB_SLEEP_EDF_DIR or BGB_DATA_ROOT and pass --run-data."
        )
    return dir_path


@pytest.fixture
def sleep_edf_raw_cropped(sleep_edf_path, mne_mod):
    """Load Sleep-EDF file cropped to 60 seconds for fast tests."""
    mne = mne_mod
    raw = mne.io.read_raw_edf(sleep_edf_path, preload=True)
    raw.crop(tmax=60)  # 1-minute slice for CI speed
    yield raw
    # Cleanup: explicitly delete to free memory
    del raw._data
    del raw


@pytest.fixture
def sleep_edf_raw_full(sleep_edf_path, mne_mod):
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

    import mne

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info)


@pytest.fixture
def mock_qc_controller():
    """Mock QC controller with proper spec and expected behavior."""
    from brain_go_brrr.domain.quality.controller import EEGQualityController

    controller = MagicMock(spec=EEGQualityController)
    controller.eegpt_model = MagicMock()  # Model is loaded
    # Set a safe default that results in ROUTINE flag for tests that don't override
    controller.run_full_qc_pipeline = MagicMock(
        return_value={
            "quality_metrics": {
                "bad_channels": [],
                "bad_channel_ratio": 0.0,
                "abnormality_score": 0.2,  # Low score for ROUTINE
                "quality_grade": "GOOD",
                "total_channels": 19,
                "artifact_ratio": 0.05,
            },
            "processing_info": {"confidence": 0.95},
            "processing_time": 0.5,
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
        return [patch("brain_go_brrr.api.routers.qc.qc_controller", mock_qc_controller)]

    return _patch


@pytest.fixture
def mock_abnormality_detector():
    """Mock abnormality detector with proper spec."""
    from brain_go_brrr.domain.abnormal.detector import AbnormalityDetector

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
