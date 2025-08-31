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
        assert req.analysis_type == "comprehensive"  # default

        # With different analysis type
        req = AnalysisRequest(
            file_path="/test.edf",
            analysis_type="qc"
        )
        assert req.analysis_type == "qc"
        assert req.file_path == "/test.edf"

    def test_qc_response_schema(self):
        """Test QCResponse schema."""
        from brain_go_brrr.api.schemas import QCResponse

        response = QCResponse(
            flag="pass",
            confidence=0.85,
            bad_channels=["Fp1", "O2"],
            quality_metrics={"snr": 10.5, "line_noise": 0.1},
            recommendation="Proceed with analysis",
            processing_time=2.5,
            quality_grade="A",
            timestamp="2024-01-01T00:00:00"
        )

        assert len(response.bad_channels) == 2
        assert response.confidence == 0.85
        assert response.flag == "pass"
        assert response.processing_time == 2.5
        assert response.quality_grade == "A"

    def test_sleep_analysis_response(self):
        """Test SleepAnalysisResponse schema."""
        from brain_go_brrr.api.schemas import SleepAnalysisResponse

        response = SleepAnalysisResponse(
            status="completed",
            sleep_stages={"W": 0.2, "N1": 0.1, "N2": 0.4, "N3": 0.2, "REM": 0.1},
            hypnogram=[{"epoch": 0, "stage": "W"}, {"epoch": 1, "stage": "N1"}],
            sleep_metrics={
                "total_sleep_time": 420,
                "sleep_efficiency": 0.85,
                "rem_percentage": 0.20
            },
            metadata={"recording_duration": 480},
            processing_time=2.5,
            timestamp="2024-01-01T00:00:00"
        )

        assert len(response.sleep_stages) == 5
        assert response.sleep_metrics["sleep_efficiency"] == 0.85
        assert response.status == "completed"


class TestAPIDependencies:
    """Test dependency injection functions."""

    def test_get_cache(self):
        """Test cache dependency can be imported."""
        from brain_go_brrr.api.dependencies import get_cache

        # Test that it's callable
        assert callable(get_cache)

    def test_get_job_store(self):
        """Test job store dependency can be imported."""
        from brain_go_brrr.api.dependencies import get_job_store

        # Test that it's callable
        assert callable(get_job_store)


class TestAPIModels:
    """Test API model classes."""

    def test_job_model(self):
        """Test Job model."""
        from datetime import datetime

        from brain_go_brrr.api.models import Job
        from brain_go_brrr.api.schemas import JobPriority, JobStatus

        now = datetime.now()
        job = Job(
            job_id="test-123",
            analysis_type="qc",
            file_path="/test.edf",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=now,
            updated_at=now
        )

        assert job.job_id == "test-123"
        assert job.status == JobStatus.PENDING
        assert job.file_path == "/test.edf"


class TestAPICache:
    """Test cache utilities."""

    def test_redis_cache_import(self):
        """Test that RedisCache can be imported."""
        from brain_go_brrr.api.cache import RedisCache

        # Test that it's a class
        assert isinstance(RedisCache, type)


class TestAPISettings:
    """Test API settings configuration."""

    def test_settings_import(self):
        """Test that APISettings can be imported."""
        from brain_go_brrr.api.settings import APISettings

        # Test that it's a class
        assert isinstance(APISettings, type)

        # Test instantiation
        settings = APISettings()
        assert settings is not None


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
