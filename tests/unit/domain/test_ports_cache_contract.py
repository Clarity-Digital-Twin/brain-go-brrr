"""Test cache port contract implementation requirements."""

import pytest


class MinimalCacheImpl:
    """Minimal implementation to test the protocol contract."""

    def get(self, key: str):
        raise NotImplementedError("get not implemented")

    def set(self, key: str, value, ttl=None):
        raise NotImplementedError("set not implemented")

    def delete(self, key: str):
        raise NotImplementedError("delete not implemented")

    def exists(self, key: str):
        raise NotImplementedError("exists not implemented")

    def clear(self):
        raise NotImplementedError("clear not implemented")


def test_cache_port_contract_requires_methods():
    """Test that CachePort protocol requires all methods."""
    cache = MinimalCacheImpl()

    # Verify it satisfies the protocol type
    assert hasattr(cache, 'get')
    assert hasattr(cache, 'set')
    assert hasattr(cache, 'delete')
    assert hasattr(cache, 'exists')

    # Test that unimplemented methods raise
    with pytest.raises(NotImplementedError):
        cache.get("k")

    with pytest.raises(NotImplementedError):
        cache.set("k", b"v", ttl=None)

    with pytest.raises(NotImplementedError):
        cache.delete("k")

    with pytest.raises(NotImplementedError):
        cache.exists("k")
