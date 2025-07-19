"""Benchmark-specific test configuration.

This module provides fixtures and configuration specifically for benchmark tests,
separate from the main test configuration.
"""

import os

import pytest

from brain_go_brrr.preprocessing.eeg_preprocessor import EEGPreprocessor


def channel_complexity_budget(n_channels: int, base_ms: float = 50.0, factor: float = 1.0) -> float:
    """Calculate performance budget based on channel count.

    EEGPT uses self-attention which scales O(n²) with channel count.
    This function provides adaptive performance targets based on complexity.

    Args:
        n_channels: Number of EEG channels
        base_ms: Base time for 19-channel standard montage
        factor: Scaling factor (can be overridden via CLI)

    Returns:
        Time budget in milliseconds

    References:
        EEGPT paper shows attention complexity scales with O(n_channels²)
    """
    # EEGPT attention complexity scales quadratically with channels
    # Standard 19-channel montage is our baseline
    complexity_ratio = (n_channels / 19) ** 2
    return factor * base_ms * complexity_ratio


@pytest.fixture(autouse=True, scope="session")
def disable_autoreject_for_benchmarks():
    """Disable Autoreject during benchmark tests for consistent timing.

    Autoreject uses cross-validation which adds variability to benchmark timings.
    This fixture ensures benchmarks measure only the core processing time.
    """
    if os.environ.get("PYTEST_CURRENT_TEST", "").startswith("tests/benchmarks/"):
        # Store original default
        original_default = EEGPreprocessor.__init__.__defaults__

        # Modify the default for use_autoreject to False
        # The defaults are a tuple, so we need to recreate it
        defaults = list(original_default) if original_default else []
        if len(defaults) >= 7:  # use_autoreject is the 7th parameter
            defaults[6] = False
            EEGPreprocessor.__init__.__defaults__ = tuple(defaults)

        yield

        # Restore original defaults
        EEGPreprocessor.__init__.__defaults__ = original_default
    else:
        yield


def pytest_addoption(parser):
    """Add benchmark-specific command line options."""
    parser.addoption(
        "--perf-budget-factor",
        action="store",
        default="1.0",
        type=float,
        help="Performance budget scaling factor (default: 1.0)",
    )


@pytest.fixture(scope="session")
def perf_budget_factor(request) -> float:
    """Get performance budget factor from CLI or default."""
    return float(request.config.getoption("--perf-budget-factor"))
