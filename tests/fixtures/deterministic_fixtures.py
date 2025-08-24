"""Centralized test fixtures for deterministic testing.

This module provides session-scoped fixtures that ensure deterministic
behavior across test runs, especially for tests involving randomness or ML models.
"""

import random
from collections.abc import Generator

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def deterministic_seed() -> Generator[int, None, None]:
    """Set global seeds for deterministic test runs.

    This fixture runs once per test session and sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch

    The seed is fixed at 42 for reproducibility.
    """
    seed = 42

    # Store original states
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    torch_cuda_available = torch.cuda.is_available()
    if torch_cuda_available:
        torch_cuda_state = torch.cuda.get_rng_state()

    # Set deterministic seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set PyTorch to deterministic mode (may impact performance)
    torch.use_deterministic_algorithms(False)  # False to avoid CUDA issues

    yield seed

    # Restore original states
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if torch_cuda_available:
        torch.cuda.set_rng_state(torch_cuda_state)


@pytest.fixture(scope="function")
def rng() -> np.random.Generator:
    """Provide a seeded random number generator for individual tests.

    This is function-scoped so each test gets a fresh RNG state.
    """
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="function")
def torch_generator() -> torch.Generator:
    """Provide a seeded PyTorch generator for individual tests."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return gen


@pytest.fixture(scope="session")
def mock_eeg_channels() -> list[str]:
    """Standard 19-channel EEG montage for testing."""
    return [
        'FP1',
        'FP2',
        'F7',
        'F3',
        'FZ',
        'F4',
        'F8',
        'T7',
        'C3',
        'CZ',
        'C4',
        'T8',
        'P7',
        'P3',
        'PZ',
        'P4',
        'P8',
        'O1',
        'O2',
    ]


@pytest.fixture(scope="session")
def old_channel_names() -> list[str]:
    """Old TUAB channel naming for compatibility testing."""
    return [
        'FP1',
        'FP2',
        'F7',
        'F3',
        'FZ',
        'F4',
        'F8',
        'T3',
        'C3',
        'CZ',
        'C4',
        'T4',  # T3/T4 instead of T7/T8
        'T5',
        'P3',
        'PZ',
        'P4',
        'T6',  # T5/T6 instead of P7/P8
        'O1',
        'O2',
    ]


@pytest.fixture(scope="function")
def mock_eeg_data(rng: np.random.Generator) -> np.ndarray:
    """Generate mock EEG data with realistic characteristics.

    Returns:
        Array of shape (19, 1024) with values in microvolts
    """
    n_channels = 19
    n_samples = 1024  # 4 seconds at 256 Hz

    # Generate base signal with 1/f characteristics
    freqs = np.fft.fftfreq(n_samples, 1 / 256)
    spectrum = np.zeros(n_samples, dtype=complex)

    # Create 1/f spectrum
    positive_freqs = freqs > 0
    spectrum[positive_freqs] = rng.standard_normal(positive_freqs.sum()) + 1j * rng.standard_normal(
        positive_freqs.sum()
    )
    spectrum[positive_freqs] /= np.sqrt(freqs[positive_freqs])

    # Generate signals
    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Random phase shift per channel
        phase_shift = rng.uniform(0, 2 * np.pi)
        shifted_spectrum = spectrum * np.exp(1j * phase_shift)

        # Convert to time domain
        signal = np.fft.ifft(shifted_spectrum).real

        # Scale to realistic EEG amplitude (10-50 µV)
        signal = signal * 30e-6 / signal.std()

        data[ch] = signal

    return data.astype(np.float32)
