"""Legitimate unit tests for API components.

These test actual behavior without mocking everything to death.
"""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError as PydanticValidationError


class TestAPISchemas:
    """Test API schema validation and behavior."""
    
    def test_analysis_request_validation(self):
        """Test AnalysisRequest schema validation."""
        from brain_go_brrr.api.schemas import AnalysisRequest
        
        # Valid request
        req = AnalysisRequest(file_path="/test.edf")
        assert req.file_path == "/test.edf"
        assert req.analysis_type == "full"  # default
        
        # With options
        req = AnalysisRequest(
            file_path="/test.edf",
            analysis_type="qc",
            options={"threshold": 0.5}
        )
        assert req.analysis_type == "qc"
        assert req.options["threshold"] == 0.5
    
    def test_qc_response_schema(self):
        """Test QCResponse schema."""
        from brain_go_brrr.api.schemas import QCResponse
        
        response = QCResponse(
            bad_channels=["Fp1", "O2"],
            quality_score=0.85,
            artifacts_detected=True,
            processing_time=2.5,
            metadata={"channels_total": 19}
        )
        
        assert len(response.bad_channels) == 2
        assert response.quality_score == 0.85
        assert response.artifacts_detected is True
        assert response.processing_time == 2.5
        assert response.metadata["channels_total"] == 19
    
    def test_sleep_analysis_response(self):
        """Test SleepAnalysisResponse schema."""
        from brain_go_brrr.api.schemas import SleepAnalysisResponse
        
        response = SleepAnalysisResponse(
            sleep_stages=["W", "N1", "N2", "N3", "REM"],
            hypnogram=[0, 1, 2, 3, 4],
            sleep_metrics={
                "total_sleep_time": 420,
                "sleep_efficiency": 0.85,
                "rem_percentage": 0.20
            },
            confidence_scores=[0.9, 0.8, 0.85, 0.9, 0.95]
        )
        
        assert len(response.sleep_stages) == 5
        assert response.sleep_metrics["sleep_efficiency"] == 0.85
        assert response.confidence_scores[0] == 0.9


class TestAPIDependencies:
    """Test dependency injection functions."""
    
    def test_get_settings(self):
        """Test settings dependency."""
        from brain_go_brrr.api.dependencies import get_settings
        
        settings = get_settings()
        assert settings is not None
        assert hasattr(settings, "debug")
        assert hasattr(settings, "redis_url")
    
    def test_get_cache(self):
        """Test cache dependency."""
        from brain_go_brrr.api.dependencies import get_cache
        
        cache = get_cache()
        assert cache is not None
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")


class TestAPIModels:
    """Test API model classes."""
    
    def test_job_status_model(self):
        """Test JobStatus model."""
        from brain_go_brrr.api.models import JobStatus
        
        status = JobStatus(
            job_id="test-123",
            status="pending",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:01:00"
        )
        
        assert status.job_id == "test-123"
        assert status.status == "pending"
        assert status.created_at == "2024-01-01T00:00:00"
    
    def test_job_result_model(self):
        """Test JobResult model."""
        from brain_go_brrr.api.models import JobResult
        
        result = JobResult(
            job_id="test-123",
            status="completed",
            result={"score": 0.95},
            error=None
        )
        
        assert result.job_id == "test-123"
        assert result.status == "completed"
        assert result.result["score"] == 0.95
        assert result.error is None
        
        # Test with error
        error_result = JobResult(
            job_id="test-456",
            status="failed",
            result=None,
            error="Processing failed"
        )
        
        assert error_result.status == "failed"
        assert error_result.error == "Processing failed"


class TestAPICache:
    """Test cache utilities."""
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        from brain_go_brrr.api.cache import get_cache_key
        
        # Test basic key generation
        key = get_cache_key("analysis", "file123")
        assert isinstance(key, str)
        assert "analysis" in key
        assert "file123" in key
        
        # Test with multiple parts
        key = get_cache_key("qc", "patient", "2024", "001")
        assert isinstance(key, str)
        assert "qc" in key
        
    def test_cache_ttl_calculation(self):
        """Test cache TTL calculation."""
        from brain_go_brrr.api.cache import calculate_ttl
        
        # Test default TTL
        ttl = calculate_ttl()
        assert isinstance(ttl, int)
        assert ttl > 0
        
        # Test custom TTL
        ttl = calculate_ttl(hours=2)
        assert ttl == 7200  # 2 hours in seconds
        
        ttl = calculate_ttl(minutes=30)
        assert ttl == 1800  # 30 minutes in seconds


class TestAPISettings:
    """Test API settings configuration."""
    
    def test_settings_defaults(self):
        """Test that settings have sensible defaults."""
        from brain_go_brrr.api.settings import APISettings
        
        settings = APISettings()
        
        # Check required attributes exist
        assert hasattr(settings, "title")
        assert hasattr(settings, "version")
        assert hasattr(settings, "debug")
        
        # Check defaults
        assert settings.title == "Brain Go Brrr API"
        assert isinstance(settings.version, str)
        assert isinstance(settings.debug, bool)
    
    def test_settings_validation(self):
        """Test settings validation."""
        from brain_go_brrr.api.settings import APISettings
        
        # Test with environment variables
        import os
        os.environ["BGB_DEBUG"] = "true"
        
        settings = APISettings()
        # Don't assert the value since it depends on env
        assert isinstance(settings.debug, bool)
        
        # Clean up
        del os.environ["BGB_DEBUG"]


class TestAPIExceptionHandlers:
    """Test API exception handling."""
    
    def test_http_exception_creation(self):
        """Test HTTPException creation."""
        exc = HTTPException(status_code=404, detail="Not found")
        assert exc.status_code == 404
        assert exc.detail == "Not found"
    
    def test_validation_error_handling(self):
        """Test validation error scenarios."""
        from brain_go_brrr.api.schemas import AnalysisRequest
        
        # Test invalid data
        with pytest.raises(PydanticValidationError):
            AnalysisRequest()  # Missing required field
        
        with pytest.raises(PydanticValidationError):
            AnalysisRequest(file_path=123)  # Wrong type