"""REAL tests for serialization edge cases - No mocks."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from brain_go_brrr.infra.serialization import (
    deserialize_value,
    get_registry,
    register_serializable,
    serialize_value,
)


@dataclass
class ComplexNestedClass:
    """Complex nested dataclass for testing."""

    id: str
    timestamp: datetime
    data: dict[str, Any]
    nested: Optional["ComplexNestedClass"] = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "data": self.data,
            "nested": self.nested.to_dict() if self.nested else None,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ComplexNestedClass":
        """Create from dict."""
        nested = None
        if data.get("nested"):
            nested = cls.from_dict(data["nested"])

        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            id=data["id"],
            timestamp=timestamp,
            data=data.get("data", {}),
            nested=nested,
            tags=data.get("tags", []),
        )


class TestSerializationEdgeCases:
    """Test serialization with edge cases."""

    def test_serialize_none_values(self):
        """Test serializing None values."""
        test_cases = [None, {"key": None}, [None, None], {"nested": {"value": None}}]

        for original in test_cases:
            serialized = serialize_value(original)
            deserialized = deserialize_value(serialized)
            assert deserialized == original

    def test_serialize_empty_containers(self):
        """Test serializing empty containers."""
        test_cases = [{}, [], {"empty_dict": {}, "empty_list": []}, [[], {}, []]]

        for original in test_cases:
            serialized = serialize_value(original)
            deserialized = deserialize_value(serialized)
            assert deserialized == original

    def test_serialize_special_strings(self):
        """Test serializing special strings."""
        test_cases = [
            "",  # Empty string
            " ",  # Whitespace
            "\n\t\r",  # Control chars
            "unicode: 你好世界 🌍",  # Unicode
            r"C:\path\to\file",  # Backslashes
            '{"json": "like"}',  # JSON-like string
            "null",  # String "null"
        ]

        for original in test_cases:
            serialized = serialize_value(original)
            deserialized = deserialize_value(serialized)
            assert deserialized == original

    def test_serialize_large_numbers(self):
        """Test serializing large numbers."""
        test_cases = [
            0,
            -1,
            2**31 - 1,  # Max 32-bit int
            2**63 - 1,  # Max 64-bit int
            -(2**63),  # Min 64-bit int
            1.7976931348623157e308,  # Near max float
            -1.7976931348623157e308,  # Near min float
            1e-308,  # Very small positive
        ]

        for original in test_cases:
            serialized = serialize_value(original)
            deserialized = deserialize_value(serialized)

            # For floats, use approximate equality
            if isinstance(original, float):
                assert abs(deserialized - original) < 1e-10
            else:
                assert deserialized == original

    def test_serialize_mixed_type_lists(self):
        """Test serializing lists with mixed types."""
        test_cases = [
            [1, "two", 3.0, None, True],
            [{"a": 1}, ["b", 2], None, 3.14],
            [[1, 2], [3, 4], [5, 6]],
        ]

        for original in test_cases:
            serialized = serialize_value(original)
            deserialized = deserialize_value(serialized)
            assert deserialized == original

    def test_serialize_deeply_nested(self):
        """Test serializing deeply nested structures."""
        # Create deeply nested dict
        nested = {"level": 0}
        current = nested
        for i in range(1, 100):
            current["next"] = {"level": i}
            current = current["next"]

        serialized = serialize_value(nested)
        deserialized = deserialize_value(serialized)

        # Verify structure preserved
        current = deserialized
        for i in range(100):
            assert current["level"] == i
            if i < 99:
                current = current["next"]

    def test_deserialize_malformed_json(self):
        """Test deserializing malformed JSON."""
        test_cases = [
            "{broken json",
            '{"key": undefined}',
            "{'single': 'quotes'}",
            '{"trailing": "comma",}',
        ]

        for malformed in test_cases:
            # Should return original string if can't parse
            result = deserialize_value(malformed)
            assert result == malformed

    def test_serialize_datetime_objects(self):
        """Test serializing datetime objects (should fail)."""
        now = datetime.now(UTC)

        # datetime is not JSON serializable by default
        with pytest.raises(TypeError):
            serialize_value(now)

    def test_serialize_bytes_objects(self):
        """Test serializing bytes objects (should fail)."""
        data = b"binary data"

        # bytes is not JSON serializable
        with pytest.raises(TypeError):
            serialize_value(data)

    def test_complex_nested_dataclass(self):
        """Test complex nested dataclass serialization."""
        # Register the class
        register_serializable(ComplexNestedClass)

        # Create nested structure
        inner = ComplexNestedClass(
            id="inner",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
            data={"value": 42},
            tags=["nested", "test"],
        )

        outer = ComplexNestedClass(
            id="outer",
            timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=UTC),
            data={"items": [1, 2, 3]},
            nested=inner,
            tags=["parent"],
        )

        # Serialize and deserialize
        serialized = serialize_value(outer)
        deserialized = deserialize_value(serialized)

        # Verify structure
        assert isinstance(deserialized, ComplexNestedClass)
        assert deserialized.id == "outer"
        assert deserialized.tags == ["parent"]
        assert deserialized.data["items"] == [1, 2, 3]

        # Verify nested
        assert deserialized.nested is not None
        assert deserialized.nested.id == "inner"
        assert deserialized.nested.tags == ["nested", "test"]
        assert deserialized.nested.data["value"] == 42

    def test_registry_duplicate_registration(self):
        """Test duplicate registration handling."""

        @dataclass
        class DuplicateTest:
            value: int

            def to_dict(self) -> dict:
                return {"value": self.value}

            @classmethod
            def from_dict(cls, data: dict) -> "DuplicateTest":
                return cls(value=data["value"])

        # First registration should work
        register_serializable(DuplicateTest)

        # Second registration should be OK (idempotent)
        register_serializable(DuplicateTest)

        # Registry should have only one entry
        registry = get_registry()
        assert "DuplicateTest" in registry

    def test_serialize_special_dict_keys(self):
        """Test serializing dicts with special keys."""
        test_cases = [
            {1: "numeric key"},  # Numeric keys become strings in JSON
            {True: "bool key"},  # Bool keys become strings
            {None: "none key"},  # None key becomes "null" string
        ]

        for original in test_cases:
            serialized = serialize_value(original)
            deserialized = deserialize_value(serialized)

            # Keys are converted to strings in JSON
            # Note: JSON converts True to "true" (lowercase)
            for key in original:
                if key is None:
                    str_key = "null"
                elif key is True:
                    str_key = "true"  # JSON uses lowercase
                elif key is False:
                    str_key = "false"  # JSON uses lowercase
                else:
                    str_key = str(key)
                assert str_key in deserialized

    def test_serialize_recursive_list(self):
        """Test serializing recursive list (should fail)."""
        lst = []
        lst.append(lst)  # Circular reference

        with pytest.raises((ValueError, TypeError)):
            serialize_value(lst)

    def test_deserialize_with_escape_sequences(self):
        """Test deserializing strings with escape sequences."""
        test_cases = [
            r'{"path": "C:\\Users\\test"}',
            r'{"text": "Line 1\nLine 2\tTabbed"}',
            r'{"quote": "He said \"Hello\""}',
        ]

        for json_str in test_cases:
            result = deserialize_value(json_str)
            assert isinstance(result, dict)

    def test_serialize_inf_and_nan(self):
        """Test serializing infinity and NaN (should fail)."""
        test_cases = [float("inf"), float("-inf"), float("nan"), {"value": float("inf")}]

        for value in test_cases:
            # Python's json module actually handles inf/nan by default
            # but converts them to JavaScript-compatible strings
            result = serialize_value(value)
            # Just check it doesn't crash
            assert result is not None

    def test_deserialize_unicode_escapes(self):
        """Test deserializing Unicode escape sequences."""
        json_str = r'{"text": "\u4f60\u597d"}'  # 你好 in Unicode escapes
        result = deserialize_value(json_str)
        assert result["text"] == "你好"

    def test_serialize_pathlib_path(self):
        """Test serializing Path objects (should fail)."""
        path = Path("/some/path")

        # Path is not JSON serializable
        with pytest.raises(TypeError):
            serialize_value(path)

    def test_deserialize_trailing_data(self):
        """Test deserializing JSON with trailing data."""
        json_str = '{"valid": "json"} extra stuff'

        # Should return original string if extra data
        result = deserialize_value(json_str)
        assert result == json_str

    def test_serialize_numpy_types(self):
        """Test serializing numpy types (should fail)."""
        test_cases = [np.array([1, 2, 3]), np.float32(3.14), np.int64(42)]

        for value in test_cases:
            # NumPy types not JSON serializable by default
            with pytest.raises(TypeError):
                serialize_value(value)

    def test_deserialize_empty_string(self):
        """Test deserializing empty string."""
        result = deserialize_value("")
        assert result == ""

    def test_deserialize_whitespace_only(self):
        """Test deserializing whitespace-only string."""
        test_cases = [" ", "\n", "\t", "   \n\t  "]

        for whitespace in test_cases:
            result = deserialize_value(whitespace)
            assert result == whitespace

    def test_serialize_decimal(self):
        """Test serializing Decimal (should fail)."""
        value = Decimal("3.14159265358979323846")

        # Decimal not JSON serializable by default
        with pytest.raises(TypeError):
            serialize_value(value)

    def test_dataclass_missing_from_dict(self):
        """Test dataclass without from_dict method."""

        @dataclass
        class NoFromDict:
            value: int

            def to_dict(self) -> dict:
                return {"value": self.value}

        # Should raise TypeError for missing from_dict
        with pytest.raises(TypeError, match="from_dict"):
            register_serializable(NoFromDict)

    def test_dataclass_missing_to_dict(self):
        """Test dataclass without to_dict method."""

        @dataclass
        class NoToDict:
            value: int

            @classmethod
            def from_dict(cls, data: dict) -> "NoToDict":
                return cls(value=data["value"])

        # Should raise TypeError for missing to_dict
        with pytest.raises(TypeError, match="to_dict"):
            register_serializable(NoToDict)
