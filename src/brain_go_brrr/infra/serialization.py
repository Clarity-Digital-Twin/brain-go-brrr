"""Dataclass serialization registry for Redis caching.

Provides a registry pattern for handling multiple dataclass types
in a scalable way, avoiding hard-coded imports in the cache layer.
"""

import dataclasses
import json
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

T = TypeVar("T")


@runtime_checkable
class SerializableDataclass(Protocol):
    """Protocol for dataclasses that can be serialized."""

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass to dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        """Create instance from dictionary."""
        ...


# Global registry of serializable dataclasses
_SERIALIZATION_REGISTRY: dict[str, type[SerializableDataclass]] = {}


def register_serializable(cls: type[T]) -> type[T]:
    """Decorator to register a dataclass for automatic serialization.

    Usage:
        @register_serializable
        @dataclass
        class MyData:
            field: str

    The class must have to_dict() and from_dict() methods.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")

    if not hasattr(cls, "to_dict") or not hasattr(cls, "from_dict"):
        raise TypeError(f"{cls.__name__} must have to_dict() and from_dict() methods")

    # Use cast to satisfy type checker while maintaining runtime safety
    _SERIALIZATION_REGISTRY[cls.__name__] = cast("type[SerializableDataclass]", cls)
    return cast("type[T]", cls)


def serialize_value(value: Any) -> str:
    """Serialize a value for Redis storage.

    Handles:
    - Registered dataclasses (via to_dict)
    - Regular dicts
    - JSON-serializable values

    Raises:
        TypeError: If value cannot be serialized
    """
    # Early validation - only handle str, bytes, or JSON-serializable types
    if isinstance(value, bytes):
        raise TypeError("Cannot serialize bytes directly - encode to base64 string first")

    if dataclasses.is_dataclass(value) and hasattr(value, "to_dict"):
        # Serialize registered dataclass
        class_name = value.__class__.__name__
        if class_name in _SERIALIZATION_REGISTRY:
            return json.dumps({"_dataclass_type": class_name, "data": value.to_dict()})
        else:
            # Fallback for unregistered dataclasses
            return json.dumps(value.to_dict())
    elif isinstance(value, dict):
        # Regular dict
        return json.dumps(value)
    elif isinstance(value, str | int | float | bool | type(None)):
        # Primitives
        return json.dumps(value)
    else:
        # Try to serialize as-is
        return json.dumps(value)


def deserialize_value(value: Any) -> Any:
    """Deserialize a value from Redis storage.

    Returns the original Python object if possible.

    Raises:
        UnicodeDecodeError: If bytes contain non-UTF8 data
    """
    # Type guard - if not str or bytes, should not happen but return as-is
    if not isinstance(value, str | bytes):
        return value  # pragma: no cover

    try:
        # Handle bytes explicitly
        if isinstance(value, bytes):
            value = value.decode("utf-8")

        decoded = json.loads(value)

        # Check if it's a registered dataclass
        if isinstance(decoded, dict) and "_dataclass_type" in decoded:
            class_name = decoded["_dataclass_type"]
            if class_name in _SERIALIZATION_REGISTRY:
                cls = _SERIALIZATION_REGISTRY[class_name]
                return cls.from_dict(decoded["data"])
            else:
                # Unknown dataclass type - return the raw dict
                return decoded["data"]

        return decoded

    except json.JSONDecodeError:
        # Not JSON, return as-is
        return value
    except UnicodeDecodeError as e:
        # Non-UTF8 bytes - re-raise with clear message
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end, f"Cannot decode bytes as UTF-8: {e.reason}"
        ) from e
    except KeyError:
        # Missing expected keys in dataclass format
        return value


def get_registry() -> dict[str, type[SerializableDataclass]]:
    """Get the current serialization registry."""
    return _SERIALIZATION_REGISTRY.copy()


def clear_registry() -> None:
    """Clear the serialization registry (for testing)."""
    _SERIALIZATION_REGISTRY.clear()
