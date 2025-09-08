"""Test domain constants are properly defined and accessible."""

from brain_go_brrr.domain.constants import (
    EEGPT_PROBE_INPUT_DIM,
    EEGPT_SUMMARY_TOKENS,
    EEGPT_TOKEN_DIM,
)


def test_eegpt_constants_values():
    """Test EEGPT constants have correct values."""
    assert EEGPT_TOKEN_DIM == 512
    assert EEGPT_SUMMARY_TOKENS == 4
    assert EEGPT_PROBE_INPUT_DIM == 2048


def test_eegpt_probe_dim_calculation():
    """Test probe dimension is correctly calculated from components."""
    assert EEGPT_PROBE_INPUT_DIM == EEGPT_SUMMARY_TOKENS * EEGPT_TOKEN_DIM
