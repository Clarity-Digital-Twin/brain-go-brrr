"""Shared test fixtures and configuration."""

from pathlib import Path

import mne
import pytest
from fastapi.testclient import TestClient

# Import benchmark fixtures to make them available
pytest_plugins = ["tests.fixtures.benchmark_data"]


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
    import api.main

    return TestClient(api.main.app)


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
    import numpy as np

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
