"""Test dataclass serialization registry functionality."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from brain_go_brrr.infra.serialization import (
    clear_registry,
    deserialize_value,
    get_registry,
    register_serializable,
    serialize_value,
)


@dataclass
class TestData1:
    """Test dataclass with basic fields."""

    name: str
    value: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"name": self.name, "value": self.value, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestData1":
        """Create from dictionary."""
        return cls(name=data["name"], value=data["value"], metadata=data.get("metadata", {}))


@dataclass
class TestData2:
    """Another test dataclass to verify multiple registrations."""

    id: str
    items: list[str] = field(default_factory=list)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"id": self.id, "items": self.items, "active": self.active}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestData2":
        """Create from dictionary."""
        return cls(id=data["id"], items=data.get("items", []), active=data.get("active", True))


@dataclass
class InvalidDataclass:
    """Dataclass without required methods."""

    value: str


class TestSerializationRegistry:
    """Test serialization registry functionality."""

    def setup_method(self) -> None:
        """Clear registry before each test."""
        clear_registry()

    def test_register_valid_dataclass(self) -> None:
        """Test registering a valid dataclass."""
        # Given: A valid dataclass with required methods
        # When: Registering it
        result = register_serializable(TestData1)

        # Then: Should return the class and be in registry
        assert result is TestData1
        registry = get_registry()
        assert "TestData1" in registry
        assert registry["TestData1"] is TestData1

    def test_register_invalid_dataclass(self) -> None:
        """Test registering invalid dataclass raises error."""

        # Given: A regular class (not dataclass)
        class NotADataclass:
            pass

        # When/Then: Should raise TypeError
        with pytest.raises(TypeError, match="must be a dataclass"):
            register_serializable(NotADataclass)

    def test_register_dataclass_without_methods(self) -> None:
        """Test registering dataclass without required methods."""
        # Given: A dataclass without to_dict/from_dict
        # When/Then: Should raise TypeError
        with pytest.raises(TypeError, match="must have to_dict"):
            register_serializable(InvalidDataclass)

    def test_round_trip_serialization_single_class(self) -> None:
        """Test round-trip serialization of a single registered class."""
        # Given: A registered dataclass instance
        register_serializable(TestData1)
        original = TestData1(name="test", value=42, metadata={"key": "value"})

        # When: Serializing and deserializing
        serialized = serialize_value(original)
        deserialized = deserialize_value(serialized)

        # Then: Should recover the same data
        assert isinstance(deserialized, TestData1)
        assert deserialized.name == original.name
        assert deserialized.value == original.value
        assert deserialized.metadata == original.metadata

    def test_round_trip_multiple_classes(self) -> None:
        """Test round-trip with multiple registered classes."""
        # Given: Multiple registered dataclasses
        register_serializable(TestData1)
        register_serializable(TestData2)

        data1 = TestData1(name="first", value=1)
        data2 = TestData2(id="abc123", items=["a", "b", "c"])

        # When: Serializing and deserializing both
        ser1 = serialize_value(data1)
        ser2 = serialize_value(data2)
        deser1 = deserialize_value(ser1)
        deser2 = deserialize_value(ser2)

        # Then: Each should deserialize to correct type
        assert isinstance(deser1, TestData1)
        assert deser1.name == "first"
        assert deser1.value == 1

        assert isinstance(deser2, TestData2)
        assert deser2.id == "abc123"
        assert deser2.items == ["a", "b", "c"]
        assert deser2.active is True

    def test_serialize_unregistered_dataclass(self) -> None:
        """Test serializing unregistered dataclass falls back gracefully."""
        # Given: An unregistered dataclass with to_dict
        data = TestData1(name="unregistered", value=99)

        # When: Serializing without registration
        serialized = serialize_value(data)
        deserialized = deserialize_value(serialized)

        # Then: Should get dict back (not instance)
        assert isinstance(deserialized, dict)
        assert deserialized["name"] == "unregistered"
        assert deserialized["value"] == 99

    def test_deserialize_unregistered_class_passthrough(self) -> None:
        """Test deserializing data for unregistered class passes through unchanged."""
        # Given: JSON data with our format but unregistered class
        unregistered_json = '{"_dataclass_type": "UnregisteredClass", "data": {"field": "value"}}'

        # When: Deserializing
        result = deserialize_value(unregistered_json)

        # Then: Should return parsed dict unchanged (no error, no instance creation)
        assert isinstance(result, dict)
        assert result["_dataclass_type"] == "UnregisteredClass"
        assert result["data"]["field"] == "value"

    def test_serialize_regular_types(self) -> None:
        """Test serialization of non-dataclass types."""
        # Test various types
        test_cases = [
            {"key": "value"},  # dict
            "hello world",  # str
            42,  # int
            3.14,  # float
            True,  # bool
            None,  # None
            ["a", "b", "c"],  # list
        ]

        for original in test_cases:
            serialized = serialize_value(original)
            deserialized = deserialize_value(serialized)
            assert deserialized == original

    def test_deserialize_invalid_json(self) -> None:
        """Test deserializing invalid JSON returns as-is."""
        # Given: Invalid JSON strings
        invalid_inputs = [
            "not json",
            b"binary data",
            "{invalid json}",
            "",
        ]

        for input_val in invalid_inputs:
            result = deserialize_value(input_val)
            assert result == input_val

    def test_registry_isolation(self) -> None:
        """Test registry can be cleared and reused."""
        # Given: A populated registry
        register_serializable(TestData1)
        assert len(get_registry()) == 1

        # When: Clearing registry
        clear_registry()

        # Then: Registry should be empty
        assert len(get_registry()) == 0

        # And: Can register again
        register_serializable(TestData2)
        assert len(get_registry()) == 1
        assert "TestData2" in get_registry()

    def test_complex_nested_data(self) -> None:
        """Test serialization of dataclass with nested complex data."""
        # Given: A dataclass with nested structures
        register_serializable(TestData1)
        complex_data = TestData1(
            name="complex",
            value=100,
            metadata={
                "nested": {"deeply": {"nested": ["values", 1, 2, 3]}},
                "lists": [[1, 2], [3, 4]],
                "mixed": [{"a": 1}, {"b": 2}],
            },
        )

        # When: Round-trip serialization
        serialized = serialize_value(complex_data)
        deserialized = deserialize_value(serialized)

        # Then: All nested data should be preserved
        assert isinstance(deserialized, TestData1)
        assert deserialized.metadata["nested"]["deeply"]["nested"] == ["values", 1, 2, 3]
        assert deserialized.metadata["lists"] == [[1, 2], [3, 4]]
        assert deserialized.metadata["mixed"] == [{"a": 1}, {"b": 2}]
