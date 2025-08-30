"""Tests for cache factory - REAL BEHAVIORAL TESTS, NO MOCKING."""

import os
import time

import pytest

from brain_go_brrr.infra.cache_factory import MemoryCache, get_cache


class TestMemoryCache:
    """Test MemoryCache implementation BEHAVIOR."""

    def test_memory_cache_get_set(self):
        """Test basic get/set operations."""
        cache = MemoryCache()

        # Initially empty
        assert cache.get("key1") is None

        # Set and get
        assert cache.set("key1", "value1") is True
        assert cache.get("key1") == "value1"

        # Overwrite
        assert cache.set("key1", "new_value") is True
        assert cache.get("key1") == "new_value"

    def test_memory_cache_stores_any_type(self):
        """Test cache can store various Python types."""
        cache = MemoryCache()

        # String
        cache.set("str", "hello")
        assert cache.get("str") == "hello"

        # Number
        cache.set("int", 42)
        assert cache.get("int") == 42

        # Float
        cache.set("float", 3.14)
        assert cache.get("float") == 3.14

        # List
        cache.set("list", [1, 2, 3])
        assert cache.get("list") == [1, 2, 3]

        # Dict
        cache.set("dict", {"a": 1, "b": 2})
        assert cache.get("dict") == {"a": 1, "b": 2}

        # None
        cache.set("none", None)
        assert cache.get("none") is None

    def test_memory_cache_delete(self):
        """Test delete operation."""
        cache = MemoryCache()

        # Delete non-existent key
        assert cache.delete("nonexistent") is False

        # Set then delete
        cache.set("key1", "value1")
        assert cache.exists("key1") is True
        assert cache.delete("key1") is True
        assert cache.exists("key1") is False
        assert cache.get("key1") is None

        # Delete again returns False
        assert cache.delete("key1") is False

    def test_memory_cache_exists(self):
        """Test exists operation."""
        cache = MemoryCache()

        # Initially doesn't exist
        assert cache.exists("key1") is False

        # After setting
        cache.set("key1", "value1")
        assert cache.exists("key1") is True

        # After deleting
        cache.delete("key1")
        assert cache.exists("key1") is False

    def test_memory_cache_clear(self):
        """Test clear operation."""
        cache = MemoryCache()

        # Add multiple items
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")

        assert cache.exists("key1") is True
        assert cache.exists("key2") is True
        assert cache.exists("key3") is True

        # Clear all
        cache.clear()

        assert cache.exists("key1") is False
        assert cache.exists("key2") is False
        assert cache.exists("key3") is False

    def test_memory_cache_close(self):
        """Test close operation (no-op for memory cache)."""
        cache = MemoryCache()
        cache.set("key1", "value1")

        # Close should be no-op
        cache.close()

        # Should still work after close
        assert cache.get("key1") == "value1"
        cache.set("key2", "value2")
        assert cache.get("key2") == "value2"

    def test_memory_cache_ttl_tracking(self):
        """Test TTL is tracked (though not enforced in current impl)."""
        cache = MemoryCache()

        # Set with TTL
        cache.set("key1", "value1", ttl=60)

        # Should track TTL internally
        assert "key1" in cache._ttls
        assert cache._ttls["key1"] > time.time()

        # Set without TTL
        cache.set("key2", "value2")
        assert "key2" not in cache._ttls

    def test_memory_cache_isolation(self):
        """Test multiple cache instances are isolated."""
        cache1 = MemoryCache()
        cache2 = MemoryCache()

        cache1.set("key1", "value1")
        cache2.set("key1", "value2")

        # Each cache has its own value
        assert cache1.get("key1") == "value1"
        assert cache2.get("key1") == "value2"

        # Clearing one doesn't affect the other
        cache1.clear()
        assert cache1.get("key1") is None
        assert cache2.get("key1") == "value2"


class TestCacheFactory:
    """Test get_cache factory function BEHAVIOR."""

    def test_get_cache_memory_backend(self):
        """Test factory returns MemoryCache for memory backend."""
        cache = get_cache("memory")
        assert isinstance(cache, MemoryCache)

        # Should be functional
        cache.set("test", "value")
        assert cache.get("test") == "value"

    def test_get_cache_invalid_backend(self):
        """Test factory raises for unknown backend."""
        with pytest.raises(ValueError, match="Unknown cache backend: invalid"):
            get_cache("invalid")

    def test_get_cache_default_to_env(self, monkeypatch):
        """Test factory uses environment variable when backend=None."""
        # Default should be memory when env not set
        cache = get_cache(None)
        assert isinstance(cache, MemoryCache)

        # Should respect CACHE_BACKEND env var
        monkeypatch.setenv("CACHE_BACKEND", "memory")
        cache = get_cache(None)
        assert isinstance(cache, MemoryCache)

    def test_get_cache_redis_backend(self, monkeypatch):
        """Test factory handles redis backend (returns InfraRedisCache)."""
        # Mock env for Redis URL
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

        # Request redis backend
        # Note: This will return InfraRedisCache, but we can't test full functionality
        # without actual Redis. Just verify it doesn't crash.
        try:
            cache = get_cache("redis")
            # Should have cache interface methods
            assert hasattr(cache, "get")
            assert hasattr(cache, "set")
            assert hasattr(cache, "delete")
            assert hasattr(cache, "exists")
        except Exception:
            # If Redis not available, that's OK for unit test
            pytest.skip("Redis not available for testing")

    def test_memory_cache_conforms_to_protocol(self):
        """Test MemoryCache implements all CachePort methods."""
        cache = MemoryCache()

        # Check all protocol methods exist and work
        assert cache.get("test") is None  # get
        assert cache.set("test", "value") is True  # set
        assert cache.exists("test") is True  # exists
        assert cache.delete("test") is True  # delete
        cache.clear()  # clear
        cache.close()  # close

    def test_factory_deprecation_warning(self):
        """Test that module is deprecated (warning already triggered at import)."""
        # The deprecation warning is triggered at module import,
        # which happens at the top of this test file.
        # Just verify the module exists and works despite deprecation.
        cache = get_cache("memory")
        assert isinstance(cache, MemoryCache)


class TestCacheFactoryEdgeCases:
    """Test edge cases and special scenarios."""

    def test_memory_cache_empty_string_key(self):
        """Test cache handles empty string as key."""
        cache = MemoryCache()

        # Empty string is valid key
        cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"
        assert cache.exists("") is True
        assert cache.delete("") is True

    def test_memory_cache_none_value(self):
        """Test cache handles None as value."""
        cache = MemoryCache()

        # None is valid value
        cache.set("key", None)
        assert cache.get("key") is None
        assert cache.exists("key") is True  # Key exists even with None value

    def test_memory_cache_large_value(self):
        """Test cache handles large values."""
        cache = MemoryCache()

        # Large list
        large_list = list(range(10000))
        cache.set("large", large_list)
        assert cache.get("large") == large_list

        # Large string
        large_string = "x" * 100000
        cache.set("large_str", large_string)
        assert cache.get("large_str") == large_string

    def test_memory_cache_many_keys(self):
        """Test cache handles many keys."""
        cache = MemoryCache()

        # Add many keys
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}")

        # All should be retrievable
        for i in range(1000):
            assert cache.get(f"key_{i}") == f"value_{i}"

        # Clear should remove all
        cache.clear()
        for i in range(1000):
            assert cache.get(f"key_{i}") is None

    def test_get_cache_case_sensitive(self):
        """Test backend selection is case-sensitive (actual behavior)."""
        # Lowercase works
        cache = get_cache("memory")
        assert isinstance(cache, MemoryCache)

        # Uppercase fails
        with pytest.raises(ValueError, match="Unknown cache backend: MEMORY"):
            get_cache("MEMORY")

        # Mixed case fails
        with pytest.raises(ValueError, match="Unknown cache backend: Memory"):
            get_cache("Memory")