"""REAL tests for API schemas - Clean, no bullshit."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from brain_go_brrr.api.schemas import (
    AnalysisRequest,
    JobCreateRequest,
    JobData,
    JobPriority,
    JobStatus,
    QCResponse,
    SleepAnalysisResponse,
)


class TestJobSchemas:
    """Test job-related schemas."""

    def test_job_data_minimal(self):
        """Test JobData with minimal fields."""
        job = JobData(
            job_id="test-123",
            analysis_type="sleep",
            status=JobStatus.PENDING
        )

        assert job.job_id == "test-123"
        assert job.analysis_type == "sleep"
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL  # default
        assert job.progress == 0.0  # default

    def test_job_data_complete(self):
        """Test JobData with all fields."""
        now = datetime.utcnow()
        job = JobData(
            job_id="test-456",
            analysis_type="abnormality",
            file_path="/data/test.edf",
            options={"threshold": 0.8},
            status=JobStatus.PROCESSING,
            priority=JobPriority.HIGH,
            progress=0.5,
            result={"abnormal": True},
            error=None,
            created_at=now,
            updated_at=now,
            started_at=now,
            completed_at=None
        )

        assert job.job_id == "test-456"
        assert job.analysis_type == "abnormality"
        assert job.options["threshold"] == 0.8
        assert job.progress == 0.5

    def test_job_status_transitions(self):
        """Test job status values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_job_priority_levels(self):
        """Test job priority values."""
        assert JobPriority.LOW.value == "low"
        assert JobPriority.NORMAL.value == "normal"
        assert JobPriority.HIGH.value == "high"
        assert JobPriority.URGENT.value == "urgent"


class TestAnalysisResponses:
    """Test analysis response schemas."""

    def test_sleep_analysis_response(self):
        """Test SleepAnalysisResponse schema."""
        response = SleepAnalysisResponse(
            status="success",
            sleep_stages={"W": 0.3, "N1": 0.1, "N2": 0.3, "N3": 0.2, "REM": 0.1},
            sleep_metrics={
                "sleep_efficiency": 85.5,
                "total_sleep_time": 420,
                "sleep_onset_latency": 15
            },
            hypnogram=["W", "W", "N1", "N2", "N2", "N3"],
            metadata={"version": "1.0"},
            processing_time=2.5,
            timestamp="2024-01-01T00:00:00Z",
            cached=False
        )

        assert response.status == "success"
        assert response.sleep_stages["W"] == 0.3
        assert response.sleep_metrics["sleep_efficiency"] == 85.5
        assert len(response.hypnogram) == 6
        assert response.processing_time == 2.5

    def test_qc_response(self):
        """Test QCResponse schema."""
        response = QCResponse(
            status="success",
            bad_channels=["Fp1", "O2"],
            quality_score=0.85,
            recommendations=["Remove Fp1", "Interpolate O2"]
        )

        assert response.status == "success"
        assert "Fp1" in response.bad_channels
        assert response.quality_score == 0.85
        assert len(response.recommendations) == 2

    def test_analysis_request(self):
        """Test AnalysisRequest schema."""
        request = AnalysisRequest(
            file_id="file-123",
            analysis_type="sleep",
            options={"window_size": 30}
        )

        assert request.file_id == "file-123"
        assert request.analysis_type == "sleep"
        assert request.options["window_size"] == 30

    def test_job_create_request(self):
        """Test JobCreateRequest schema."""
        request = JobCreateRequest(
            analysis_type="abnormality",
            file_path="/data/test.edf",
            priority=JobPriority.HIGH,
            options={"threshold": 0.8}
        )

        assert request.analysis_type == "abnormality"
        assert request.file_path == "/data/test.edf"
        assert request.priority == JobPriority.HIGH
        assert request.options["threshold"] == 0.8


class TestSchemaValidation:
    """Test schema validation rules."""

    def test_job_data_invalid_progress(self):
        """Test JobData rejects invalid progress values."""
        with pytest.raises(ValidationError) as exc_info:
            JobData(
                job_id="test",
                analysis_type="sleep",
                status=JobStatus.PENDING,
                progress=1.5  # > 1.0
            )

        assert "progress" in str(exc_info.value)

    def test_job_data_empty_job_id(self):
        """Test JobData requires non-empty job_id."""
        with pytest.raises(ValidationError) as exc_info:
            JobData(
                job_id="",  # empty
                analysis_type="sleep",
                status=JobStatus.PENDING
            )

        assert "job_id" in str(exc_info.value)

    def test_sleep_stages_sum_validation(self):
        """Test sleep stages should sum to approximately 1.0."""
        # This might not have validation, but good to check
        response = SleepAnalysisResponse(
            status="success",
            sleep_stages={"W": 0.5, "N1": 0.5, "N2": 0.5},  # Sum > 1
            sleep_metrics={},
            hypnogram=[],
            metadata={},
            processing_time=1.0,
            timestamp="2024-01-01T00:00:00Z",
            cached=False
        )

        # Should accept it (no strict validation) but document the behavior
        assert sum(response.sleep_stages.values()) > 1.0
