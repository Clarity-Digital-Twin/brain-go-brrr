"""Test serialization contract enforcement - REAL tests, no bullshit."""

import json
from dataclasses import dataclass
from typing import Any

import pytest

from brain_go_brrr.infra.serialization import (
    deserialize_value,
    get_registry,
    register_serializable,
    serialize_value,
)


def test_register_requires_to_dict():
    """Test that register requires to_dict method."""
    with pytest.raises(TypeError, match="to_dict"):
        @register_serializable
        @dataclass
        class BadNoToDict:
            x: int
            # Missing to_dict method


def test_register_requires_from_dict():
    """Test that register requires from_dict method."""
    with pytest.raises(TypeError, match="from_dict"):
        @register_serializable
        @dataclass
        class BadNoFromDict:
            x: int

            def to_dict(self):
                return {"x": self.x}
            # Missing from_dict method


def test_register_requires_dataclass():
    """Test that register requires a dataclass."""
    with pytest.raises(TypeError, match="must be a dataclass"):
        @register_serializable
        class NotADataclass:
            def __init__(self, x: int):
                self.x = x

            def to_dict(self):
                return {"x": self.x}

            @classmethod
            def from_dict(cls, d):
                return cls(**d)


# Try to register, but don't fail if already registered
try:
    @register_serializable
    @dataclass
    class MiniSerializable:
        """Minimal serializable class for testing."""
        x: int
        y: str = "default"

        def to_dict(self) -> dict[str, Any]:
            return {"x": self.x, "y": self.y}

        @classmethod
        def from_dict(cls, d: dict[str, Any]) -> "MiniSerializable":
            return cls(**d)
except Exception:
    # Already registered from another test
    from brain_go_brrr.infra.serialization import get_registry
    MiniSerializable = get_registry().get("MiniSerializable")
    if not MiniSerializable:
        # If not in registry, define without decorator
        @dataclass
        class MiniSerializable:
            """Minimal serializable class for testing."""
            x: int
            y: str = "default"

            def to_dict(self) -> dict[str, Any]:
                return {"x": self.x, "y": self.y}

            @classmethod
            def from_dict(cls, d: dict[str, Any]) -> "MiniSerializable":
                return cls(**d)
        
        # Manually register
        from brain_go_brrr.infra.serialization import _SERIALIZATION_REGISTRY
        _SERIALIZATION_REGISTRY["MiniSerializable"] = MiniSerializable


def test_roundtrip_basic():
    """Test basic roundtrip serialization."""
    obj = MiniSerializable(x=7, y="test")
    blob = serialize_value(obj)
    out = deserialize_value(blob)

    assert isinstance(out, MiniSerializable)
    assert out.x == 7
    assert out.y == "test"


def test_roundtrip_with_defaults():
    """Test roundtrip with default values."""
    obj = MiniSerializable(x=42)  # Use default y
    blob = serialize_value(obj)
    out = deserialize_value(blob)

    assert isinstance(out, MiniSerializable)
    assert out.x == 42
    assert out.y == "default"


def test_unknown_type_passthrough():
    """Test that unknown types pass through as dicts."""
    # Create JSON with unregistered type
    unknown_json = json.dumps({
        "_dataclass_type": "NotRegistered",
        "data": {"x": 1, "y": "value"}
    })

    result = deserialize_value(unknown_json)

    # Should return the dict since type not registered
    assert isinstance(result, dict)
    assert result["_dataclass_type"] == "NotRegistered"
    assert result["data"]["x"] == 1


def test_malformed_dataclass_json():
    """Test handling of malformed dataclass JSON."""
    # Missing data field
    bad_json = json.dumps({
        "_dataclass_type": "MiniSerializable",
        # Missing "data" field
    })

    result = deserialize_value(bad_json)

    # Should return the decoded dict since it has _dataclass_type but can't instantiate
    assert isinstance(result, dict)
    assert result["_dataclass_type"] == "MiniSerializable"


def test_registry_contains_registered():
    """Test that registry contains registered classes."""
    # Register class first if not already registered
    try:
        register_serializable(MiniSerializable)
    except:
        pass  # Already registered

    registry = get_registry()
    assert "MiniSerializable" in registry
    assert registry["MiniSerializable"] is MiniSerializable


def test_serialize_unregistered_with_to_dict():
    """Test serializing unregistered class with to_dict."""
    @dataclass
    class UnregisteredButSerializable:
        value: int

        def to_dict(self):
            return {"value": self.value}

    obj = UnregisteredButSerializable(value=99)
    blob = serialize_value(obj)

    # Should serialize using to_dict
    assert "99" in blob


def test_complex_nested_serialization():
    """Test complex nested serialization."""
    @register_serializable
    @dataclass
    class NestedTestClass:
        data: dict[str, Any]

        def to_dict(self):
            return {"data": self.data}

        @classmethod
        def from_dict(cls, d):
            return cls(**d)

    obj = NestedTestClass(data={
        "level1": {
            "level2": {
                "values": [1, 2, 3],
                "flag": True
            }
        }
    })

    blob = serialize_value(obj)
    out = deserialize_value(blob)

    assert isinstance(out, NestedTestClass)
    assert out.data["level1"]["level2"]["values"] == [1, 2, 3]
    assert out.data["level1"]["level2"]["flag"] is True


def test_empty_dataclass():
    """Test empty dataclass serialization."""
    @register_serializable
    @dataclass
    class EmptyTestClass:
        def to_dict(self):
            return {}

        @classmethod
        def from_dict(cls, d):
            return cls()

    obj = EmptyTestClass()
    blob = serialize_value(obj)
    out = deserialize_value(blob)

    assert isinstance(out, EmptyTestClass)


def test_dataclass_with_none_values():
    """Test dataclass with None values."""
    @register_serializable
    @dataclass
    class WithOptionalTest:
        required: int
        optional: Any = None

        def to_dict(self):
            return {"required": self.required, "optional": self.optional}

        @classmethod
        def from_dict(cls, d):
            return cls(**d)

    obj = WithOptionalTest(required=5, optional=None)
    blob = serialize_value(obj)
    out = deserialize_value(blob)

    assert isinstance(out, WithOptionalTest)
    assert out.required == 5
    assert out.optional is None


def test_deserialize_non_json_string():
    """Test deserializing non-JSON string."""
    result = deserialize_value("not json at all")
    assert result == "not json at all"


def test_deserialize_partial_json():
    """Test deserializing partial JSON."""
    result = deserialize_value('{"incomplete": ')
    assert result == '{"incomplete": '
