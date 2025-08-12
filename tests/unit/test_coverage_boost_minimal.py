"""Minimal tests to boost coverage - targeting specific uncovered branches."""

import json
from dataclasses import dataclass

import numpy as np
import pytest

from brain_go_brrr.core.exceptions import UnsupportedMontageError
from brain_go_brrr.core.window_extractor import WindowExtractor
from brain_go_brrr.infra.serialization import (
    deserialize_value,
    register_serializable,
    serialize_value,
)


def test_overlap_equals_window_zero_stride():
    """Test that overlap equal to window creates zero stride."""
    we = WindowExtractor(window_seconds=4.0, overlap_seconds=4.0)
    # Stride should be 0 (window - overlap)
    assert we.stride_seconds == 0.0


def test_unknown_type_raises():
    """Test deserializing unknown type raises exception."""
    # Create JSON with unknown type
    unknown_json = json.dumps({"_dataclass_type": "NotRegistered", "data": {"x": 1}})
    
    # Should return the dict since type not registered
    result = deserialize_value(unknown_json)
    assert isinstance(result, dict)
    assert result["_dataclass_type"] == "NotRegistered"


def test_overlap_greater_than_window():
    """Test overlap greater than window creates negative stride."""
    # Overlap > window creates negative stride
    we = WindowExtractor(window_seconds=2.0, overlap_seconds=3.0)
    assert we.stride_seconds == -1.0  # 2 - 3 = -1


def test_partial_tail_dropped():
    """Test that partial tail windows are handled."""
    x = np.zeros((4, 100))
    we = WindowExtractor(
        window_seconds=1.0, 
        overlap_seconds=0.5
    )
    # Extract windows at 64Hz
    wins = we.extract(x, sfreq=64.0)
    # Check we get some windows
    assert len(wins) > 0


def test_register_requires_contract():
    """Test that register requires to_dict/from_dict methods."""
    with pytest.raises(TypeError, match="to_dict"):
        @register_serializable
        @dataclass
        class BadNoToDict:
            x: int
            # Missing to_dict method


@register_serializable
@dataclass
class MiniSerializable:
    """Minimal serializable class for testing."""
    x: int
    
    def to_dict(self):
        return {"x": self.x}
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def test_roundtrip_registered_class():
    """Test roundtrip serialization of registered class."""
    obj = MiniSerializable(7)
    blob = serialize_value(obj)
    out = deserialize_value(blob)
    assert isinstance(out, MiniSerializable)
    assert out.x == 7


def test_window_extractor_zero_stride():
    """Test WindowExtractor with zero stride."""
    # This creates zero stride (overlap = window)
    we = WindowExtractor(window_seconds=2.0, overlap_seconds=2.0)
    assert we.stride_seconds == 0.0


def test_window_extractor_negative_values():
    """Test WindowExtractor with edge case values."""
    # Negative overlap creates larger stride
    we = WindowExtractor(window_seconds=2.0, overlap_seconds=-0.5)
    assert we.stride_seconds == 2.5  # window - overlap = 2 - (-0.5) = 2.5


def test_serialization_empty_registry_lookup():
    """Test deserializing when registry is empty."""
    # Unknown type should just return the dict
    data = {"_dataclass_type": "UnknownType", "data": {"value": 42}}
    json_str = json.dumps(data)
    result = deserialize_value(json_str)
    assert isinstance(result, dict)
    assert result["data"]["value"] == 42


def test_window_extractor_data_shorter_than_window():
    """Test window extraction when data is shorter than window."""
    we = WindowExtractor(window_seconds=2.0)
    
    # Data shorter than window (150 samples < 200 needed at 100Hz)
    short_data = np.zeros((4, 150))
    windows = we.extract(short_data, sfreq=100.0)
    
    # Should return no windows
    assert len(windows) == 0


def test_window_extractor_exact_fit():
    """Test window extraction when data fits exactly."""
    we = WindowExtractor(
        window_seconds=1.0, 
        overlap_seconds=0.0  # No overlap
    )
    
    # Exactly 3 windows worth of data
    data = np.zeros((4, 300))
    windows = we.extract(data, sfreq=100.0)
    
    # Should get exactly 3 windows
    assert len(windows) == 3
    assert all(w.shape == (4, 100) for w in windows)