"""Tests for API models."""

from datetime import datetime

import pytest

from brain_go_brrr.api.models import Job
from brain_go_brrr.api.schemas import JobPriority, JobStatus


class TestJobModel:
    """Test Job data model."""

    def test_job_creation_with_required_fields(self):
        """Test creating a job with required fields."""
        now = datetime.now()
        job = Job(
            job_id="test-123",
            analysis_type="qc",
            file_path="/path/to/file.edf",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=now,
            updated_at=now,
        )

        assert job.job_id == "test-123"
        assert job.analysis_type == "qc"
        assert job.file_path == "/path/to/file.edf"
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert job.progress == 0.0
        assert job.result is None
        assert job.error is None

    def test_job_creation_with_all_fields(self):
        """Test creating a job with all fields."""
        now = datetime.now()
        job = Job(
            job_id="test-456",
            analysis_type="sleep",
            file_path="/path/to/sleep.edf",
            status=JobStatus.COMPLETED,
            priority=JobPriority.HIGH,
            created_at=now,
            updated_at=now,
            options={"include_report": True},
            progress=1.0,
            result={"sleep_efficiency": 0.85},
            error=None,
            started_at=now,
            completed_at=now,
        )

        assert job.options == {"include_report": True}
        assert job.progress == 1.0
        assert job.result == {"sleep_efficiency": 0.85}
        assert job.started_at == now
        assert job.completed_at == now

    def test_job_validation_empty_job_id(self):
        """Test job validation with empty job_id."""
        with pytest.raises(ValueError, match="job_id cannot be empty"):
            Job(
                job_id="",
                analysis_type="qc",
                file_path="/path/to/file.edf",
                status=JobStatus.PENDING,
                priority=JobPriority.NORMAL,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_job_validation_empty_analysis_type(self):
        """Test job validation with empty analysis_type."""
        with pytest.raises(ValueError, match="analysis_type cannot be empty"):
            Job(
                job_id="test-123",
                analysis_type="",
                file_path="/path/to/file.edf",
                status=JobStatus.PENDING,
                priority=JobPriority.NORMAL,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def test_job_validation_invalid_progress(self):
        """Test job validation with invalid progress."""
        with pytest.raises(ValueError, match="Progress must be between 0.0 and 1.0"):
            Job(
                job_id="test-123",
                analysis_type="qc",
                file_path="/path/to/file.edf",
                status=JobStatus.PENDING,
                priority=JobPriority.NORMAL,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                progress=1.5,
            )

    def test_job_to_dict(self):
        """Test converting job to dictionary."""
        now = datetime.now()
        job = Job(
            job_id="test-789",
            analysis_type="abnormal",
            file_path="/path/to/abnormal.edf",
            status=JobStatus.PROCESSING,
            priority=JobPriority.LOW,
            created_at=now,
            updated_at=now,
            progress=0.5,
        )

        data = job.to_dict()

        assert data["job_id"] == "test-789"
        assert data["analysis_type"] == "abnormal"
        assert data["file_path"] == "/path/to/abnormal.edf"
        assert data["status"] == JobStatus.PROCESSING.value
        assert data["priority"] == JobPriority.LOW.value
        assert data["progress"] == 0.5
        assert data["created_at"] == now.isoformat()
        assert data["updated_at"] == now.isoformat()
        assert data["started_at"] is None
        assert data["completed_at"] is None

    def test_job_from_dict(self):
        """Test creating job from dictionary."""
        now = datetime.now()
        data = {
            "job_id": "test-101",
            "analysis_type": "events",
            "file_path": "/path/to/events.edf",
            "status": "failed",
            "priority": "high",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "progress": 0.75,
            "error": "Processing failed",
        }

        job = Job.from_dict(data)

        assert job.job_id == "test-101"
        assert job.analysis_type == "events"
        assert job.file_path == "/path/to/events.edf"
        assert job.status == JobStatus.FAILED
        assert job.priority == JobPriority.HIGH
        assert job.progress == 0.75
        assert job.error == "Processing failed"

    def test_job_immutability(self):
        """Test that Job is immutable (frozen dataclass)."""
        job = Job(
            job_id="test-immutable",
            analysis_type="qc",
            file_path="/path/to/file.edf",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        with pytest.raises(AttributeError):
            job.job_id = "changed"

    def test_job_roundtrip_serialization(self):
        """Test converting job to dict and back."""
        now = datetime.now()
        original = Job(
            job_id="test-roundtrip",
            analysis_type="sleep",
            file_path="/path/to/sleep.edf",
            status=JobStatus.COMPLETED,
            priority=JobPriority.NORMAL,
            created_at=now,
            updated_at=now,
            options={"window_size": 30},
            progress=1.0,
            result={"stages": ["W", "N1", "N2", "N3", "REM"]},
            started_at=now,
            completed_at=now,
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = Job.from_dict(data)

        # Compare all fields
        assert restored.job_id == original.job_id
        assert restored.analysis_type == original.analysis_type
        assert restored.file_path == original.file_path
        assert restored.status == original.status
        assert restored.priority == original.priority
        assert restored.options == original.options
        assert restored.progress == original.progress
        assert restored.result == original.result
        # Datetime comparison might have microsecond differences due to isoformat
        assert restored.created_at.replace(microsecond=0) == original.created_at.replace(microsecond=0)
