"""Test suite for P1 fixes to ensure they work correctly."""

import pytest
from brain_go_brrr.infra.cache import InMemoryCache


class TestInMemoryCachePatternFix:
    """Test that InMemoryCache.clear_pattern works correctly after P1 fix."""
    
    def test_clear_pattern_with_wildcard(self):
        """Test that shell wildcard patterns work correctly."""
        cache = InMemoryCache()
        
        # Set up test data
        cache.set("eeg_analysis_123", "value1")
        cache.set("eeg_analysis_456", "value2")
        cache.set("eeg_result_789", "value3")
        cache.set("other_data", "value4")
        
        # Clear all eeg_analysis_* keys
        deleted = cache.clear_pattern("eeg_analysis_*")
        
        # Verify correct keys were deleted
        assert deleted == 2
        assert cache.get("eeg_analysis_123") is None
        assert cache.get("eeg_analysis_456") is None
        assert cache.get("eeg_result_789") == "value3"  # Should remain
        assert cache.get("other_data") == "value4"  # Should remain
    
    def test_clear_pattern_with_complex_pattern(self):
        """Test more complex patterns."""
        cache = InMemoryCache()
        
        # Set up test data
        cache.set("analysis:v1.0.0:abc", "value1")
        cache.set("analysis:v1.0.0:def", "value2")
        cache.set("analysis:v2.0.0:abc", "value3")
        cache.set("metrics:v1.0.0:abc", "value4")
        
        # Clear analysis:v1.0.0:* keys
        deleted = cache.clear_pattern("analysis:v1.0.0:*")
        
        assert deleted == 2
        assert cache.get("analysis:v1.0.0:abc") is None
        assert cache.get("analysis:v1.0.0:def") is None
        assert cache.get("analysis:v2.0.0:abc") == "value3"
        assert cache.get("metrics:v1.0.0:abc") == "value4"
    
    def test_clear_pattern_all(self):
        """Test clearing all keys with * pattern."""
        cache = InMemoryCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        deleted = cache.clear_pattern("*")
        
        assert deleted == 3
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        assert cache.get("key3") is None