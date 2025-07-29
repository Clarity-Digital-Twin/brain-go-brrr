"""Cache-related test fixtures."""

import hashlib
from typing import Any

import pytest
from fastapi.testclient import TestClient


class DummyCache:
    """Minimal Redis stub that records calls and maintains state."""

    def __init__(self):
        """Initialize the cache stub."""
        self.connected = True
        self._store: dict[str, Any] = {}
        self.mock_calls: list[tuple[str, ...]] = []

    def set(
        self, key: str, value: Any, *, ttl: int | None = None, expiry: int | None = None
    ) -> bool:
        """Store value and record the call."""
        self._store[key] = value
        self.mock_calls.append(("set", key, value, {"ttl": ttl, "expiry": expiry}))
        return True

    def get(self, key: str) -> Any | None:
        """Retrieve value and record the call."""
        self.mock_calls.append(("get", key))
        return self._store.get(key)

    def generate_cache_key(self, content: bytes, analysis_type: str) -> str:
        """Generate cache key from content and analysis type."""
        content_hash = hashlib.sha256(content).hexdigest()
        key = f"eeg_analysis:{content_hash}:{analysis_type}"
        self.mock_calls.append(("generate_cache_key", content, analysis_type, key))
        return key

    def exists(self, key: str) -> int:
        """Check if key exists."""
        self.mock_calls.append(("exists", key))
        return 1 if key in self._store else 0

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiry (no-op in tests)."""
        self.mock_calls.append(("expire", key, seconds))
        return key in self._store

    def delete(self, *keys: str) -> int:
        """Delete keys."""
        self.mock_calls.append(("delete", *keys))
        count = 0
        for key in keys:
            if key in self._store:
                del self._store[key]
                count += 1
        return count

    def ping(self) -> bool:
        """Check connection."""
        self.mock_calls.append(("ping",))
        return True

    def keys(self, pattern: str = "*") -> list[str]:
        """List keys matching pattern.

        Simple pattern matching for tests.
        """
        self.mock_calls.append(("keys", pattern))
        if pattern == "*":
            return list(self._store.keys())
        # Simple pattern matching for tests
        prefix = pattern.rstrip("*")
        return [k for k in self._store if k.startswith(prefix)]

    def info(self) -> dict:
        """Get info."""
        self.mock_calls.append(("info",))
        return {"redis_version": "test", "used_memory": len(self._store)}

    def dbsize(self) -> int:
        """Get number of keys."""
        self.mock_calls.append(("dbsize",))
        return len(self._store)

    def get_stats(self) -> dict:
        """Get cache stats."""
        return {
            "connected": self.connected,
            "memory_usage": "N/A",
            "total_keys": len(self._store),
            "hit_rate": 0.0,
        }

    def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern."""
        keys_to_delete = self.keys(pattern)
        return self.delete(*keys_to_delete) if keys_to_delete else 0

    def reset_mock(self):
        """Reset recorded calls."""
        self.mock_calls = []

    def set_calls(self) -> list[tuple[str, ...]]:
        """Get all set calls."""
        return [call for call in self.mock_calls if call[0] == "set"]

    def get_calls(self) -> list[tuple[str, ...]]:
        """Get all get calls."""
        return [call for call in self.mock_calls if call[0] == "get"]


@pytest.fixture
def dummy_cache():
    """Provide a fresh cache stub for each test."""
    return DummyCache()


@pytest.fixture
def client_with_cache(dummy_cache):
    """Create test client with cache dependency properly overridden."""
    import brain_go_brrr.api.main as api_main
    from brain_go_brrr.api.cache import get_cache

    # Override the cache dependency
    api_main.app.dependency_overrides[get_cache] = lambda: dummy_cache

    with TestClient(api_main.app) as client:
        yield client

    # Clean up the override
    api_main.app.dependency_overrides.pop(get_cache, None)
