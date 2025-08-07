"""Tests for API schemas - Fixed to match ACTUAL API."""

from datetime import UTC, datetime

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

    def test_job_data_complete(self):
        """Test JobData with all fields."""
        now = datetime.now(UTC)
        job = JobData(
            job_id="test-456",
            analysis_type="abnormality",
            file_path="/data/test.edf",
            status=JobStatus.PROCESSING,
            priority=JobPriority.HIGH,
            created_at=now,
            updated_at=now,
            options={"threshold": 0.8},
            progress=0.5,
            result={"abnormal": True},
            error=None,
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
            hypnogram=[{"stage": "W", "time": 0}, {"stage": "N1", "time": 30}],
            metadata={"version": "1.0"},
            processing_time=2.5,
            timestamp="2024-01-01T00:00:00Z",
            cached=False
        )

        assert response.status == "success"
        assert response.sleep_stages["W"] == 0.3
        assert response.sleep_metrics["sleep_efficiency"] == 85.5
        assert len(response.hypnogram) == 2
        assert response.processing_time == 2.5

    def test_qc_response(self):
        """Test QCResponse schema."""
        response = QCResponse(
            flag="good",
            confidence=0.95,
            bad_channels=["Fp1", "O2"],
            quality_metrics={"snr": 10.5, "artifact_ratio": 0.1},
            recommendation="Remove Fp1",
            processing_time=1.5,
            quality_grade="A",
            timestamp="2024-01-01T00:00:00Z"
        )

        assert response.flag == "good"
        assert "Fp1" in response.bad_channels
        assert response.confidence == 0.95
        assert response.quality_grade == "A"

    def test_analysis_request(self):
        """Test AnalysisRequest schema."""
        request = AnalysisRequest(
            file_path="/data/file-123.edf",
            analysis_type="sleep"
        )

        assert request.file_path == "/data/file-123.edf"
        assert request.analysis_type == "sleep"

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
