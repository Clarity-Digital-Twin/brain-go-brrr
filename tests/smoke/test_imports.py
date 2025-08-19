"""Smoke tests for critical imports.

These tests ensure that our public API imports work correctly
and that refactorings haven't broken module exports.
"""


def test_services_imports():
    """Test that services module exports work."""
    from brain_go_brrr.services import (
        HierarchicalEEGAnalyzer,
        HierarchicalPipelineYASAAdapter,
        YASAConfig,
        YASASleepStager,
    )

    assert HierarchicalEEGAnalyzer is not None
    assert HierarchicalPipelineYASAAdapter is not None
    assert YASAConfig is not None
    assert YASASleepStager is not None


def test_preprocessing_imports():
    """Test that preprocessing exports work (both old and new paths)."""
    # New canonical path
    from brain_go_brrr.preprocessing import (
        BandpassFilter,
        PreprocessingConfig,
        PreprocessingPipeline,
    )

    assert BandpassFilter is not None
    assert PreprocessingConfig is not None
    assert PreprocessingPipeline is not None

    # Test domain channels module (replacement for core.preprocessing_utils)
    from brain_go_brrr.domain import channels
    from brain_go_brrr.domain.channels import ChannelMapper

    assert channels is not None
    assert ChannelMapper is not None


def test_core_job_models_imports():
    """Test that job models are accessible from application layer."""
    from brain_go_brrr.application.jobs.models import JobData, JobPriority, JobStatus

    assert JobData is not None
    assert JobPriority is not None
    assert JobStatus is not None

    # Test enums have expected values
    assert JobStatus.PENDING == "pending"
    assert JobStatus.RUNNING == "running"
    assert JobStatus.COMPLETED == "completed"
    assert JobStatus.FAILED == "failed"

    assert JobPriority.LOW == "low"
    assert JobPriority.NORMAL == "normal"
    assert JobPriority.HIGH == "high"
    assert JobPriority.CRITICAL == "critical"


def test_cache_port_protocol():
    """Test that cache protocol is importable from infra layer."""
    from brain_go_brrr.infra.cache import RedisCache, RedisCacheProtocol

    assert RedisCacheProtocol is not None
    assert RedisCache is not None


def test_infra_cache_factory():
    """Test that cache factory works."""
    from brain_go_brrr.infra.cache_factory import MemoryCache, get_cache

    # Test memory cache can be instantiated
    cache = MemoryCache()
    assert cache is not None

    # Test factory returns a cache
    cache_instance = get_cache(backend="memory")
    assert cache_instance is not None


def test_api_still_has_job_models():
    """Test that API schemas still work (for backwards compat)."""
    from brain_go_brrr.api.schemas import JobPriority, JobStatus

    # These should exist (even if they're re-exports)
    assert JobStatus is not None
    assert JobPriority is not None

    # API might have extra values
    assert hasattr(JobStatus, "PENDING")
    assert hasattr(JobStatus, "PROCESSING")  # API-specific alias
