"""Test fixtures for brain-go-brrr."""

from .mock_eegpt import (
    MockEEGPTModel,
    MockClassifierHead,
    create_mock_detector_with_realistic_model
)

__all__ = [
    "MockEEGPTModel",
    "MockClassifierHead", 
    "create_mock_detector_with_realistic_model"
]
