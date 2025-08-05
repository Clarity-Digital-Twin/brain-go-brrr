"""Tests for API dependencies."""

import asyncio

from brain_go_brrr.api import dependencies
from brain_go_brrr.api.cache import RedisCache


class TestAPIDependencies:
    """Test API dependency functions."""

    def test_get_cache_returns_none_when_not_initialized(self):
        """Test get_cache returns None when cache_client is not initialized."""
        # Save original state
        original_cache = dependencies.cache_client

        try:
            # Set cache_client to None
            dependencies.cache_client = None

            result = asyncio.run(dependencies.get_cache())
            assert result is None
        finally:
            # Restore original state
            dependencies.cache_client = original_cache

    def test_get_cache_returns_cache_when_initialized(self):
        """Test get_cache returns cache instance when initialized."""
        # Save original state
        original_cache = dependencies.cache_client

        try:
            # Create a mock cache instance
            mock_cache = RedisCache()
            dependencies.cache_client = mock_cache

            result = asyncio.run(dependencies.get_cache())
            assert result is mock_cache
        finally:
            # Restore original state
            dependencies.cache_client = original_cache

    def test_get_job_store_returns_dict(self):
        """Test get_job_store returns the job store dictionary."""
        # Save original state
        original_store = dependencies.job_store

        try:
            # Create test data
            test_store = {"job1": {"status": "running"}, "job2": {"status": "completed"}}
            dependencies.job_store = test_store

            result = asyncio.run(dependencies.get_job_store())
            assert result is test_store
            assert result["job1"]["status"] == "running"
            assert result["job2"]["status"] == "completed"
        finally:
            # Restore original state
            dependencies.job_store = original_store

    def test_get_job_store_returns_empty_dict_by_default(self):
        """Test get_job_store returns empty dict when not populated."""
        # Save original state
        original_store = dependencies.job_store

        try:
            # Reset to empty
            dependencies.job_store = {}

            result = asyncio.run(dependencies.get_job_store())
            assert result == {}
            assert isinstance(result, dict)
        finally:
            # Restore original state
            dependencies.job_store = original_store

    def test_module_level_globals_exist(self):
        """Test that module-level globals are defined."""
        assert hasattr(dependencies, "cache_client")
        assert hasattr(dependencies, "job_store")
        assert isinstance(dependencies.job_store, dict)
