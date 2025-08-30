"""Test cache port contract to cover abstract method branches."""

import pytest

from brain_go_brrr.domain.ports.cache import CachePort


def test_cache_port_contract_raises():
    """Test that abstract CachePort methods raise NotImplementedError."""
    p = CachePort()
    
    # All abstract methods should raise
    with pytest.raises(NotImplementedError):
        p.get("k")
    
    with pytest.raises(NotImplementedError):
        p.set("k", b"v", ttl=None)
    
    with pytest.raises(NotImplementedError):
        p.delete("k")
    
    with pytest.raises(NotImplementedError):
        p.exists("k")
    
    with pytest.raises(NotImplementedError):
        p.clear()