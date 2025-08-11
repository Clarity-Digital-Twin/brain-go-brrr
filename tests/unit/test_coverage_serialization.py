"""Tests for serialization module to boost coverage."""

import pytest
import json
from dataclasses import dataclass
from brain_go_brrr.infra.serialization import (
    serialize_value,
    deserialize_value,
    register_serializable,
    get_registry,
    clear_registry
)


def test_serialize_basic_types():
    """Test serialization of basic types."""
    # String
    assert serialize_value("test") == '"test"'
    
    # Number
    assert serialize_value(42) == '42'
    
    # Boolean
    assert serialize_value(True) == 'true'
    
    # None
    assert serialize_value(None) == 'null'
    
    # List
    assert serialize_value([1, 2, 3]) == '[1, 2, 3]'
    
    # Dict
    assert serialize_value({"key": "value"}) == '{"key": "value"}'


def test_deserialize_basic_types():
    """Test deserialization of basic types."""
    # Already deserialized
    assert deserialize_value(42) == 42
    assert deserialize_value("test") == "test"
    
    # JSON strings
    assert deserialize_value('{"key": "value"}') == {"key": "value"}
    assert deserialize_value('[1, 2, 3]') == [1, 2, 3]
    
    # Invalid JSON returns as-is
    assert deserialize_value("not json") == "not json"


def test_serializable_registry():
    """Test the serializable registry."""
    # Clear registry first
    clear_registry()
    
    # Initially empty
    registry = get_registry()
    initial_size = len(registry)
    
    # Register a class with required methods
    @register_serializable
    @dataclass
    class TestClass:
        value: int
        
        def to_dict(self):
            return {"value": self.value}
        
        @classmethod
        def from_dict(cls, data):
            return cls(**data)
    
    # Should be in registry
    registry = get_registry()
    assert "TestClass" in registry
    assert registry["TestClass"] == TestClass
    
    # Clear registry
    clear_registry()
    registry = get_registry()
    # Should be back to initial size or empty
    assert len(registry) <= initial_size


def test_serialize_complex_nested():
    """Test serialization of nested structures."""
    data = {
        "level1": {
            "level2": {
                "values": [1, 2, 3],
                "flag": True
            }
        }
    }
    
    serialized = serialize_value(data)
    assert isinstance(serialized, str)
    
    # Should be valid JSON
    deserialized = json.loads(serialized)
    assert deserialized == data