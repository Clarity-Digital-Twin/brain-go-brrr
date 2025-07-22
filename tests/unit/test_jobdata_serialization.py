"""Test JobData serialization for Redis caching."""

import json
from datetime import UTC, datetime

from brain_go_brrr.api.schemas import JobData, JobPriority, JobStatus


class TestJobDataSerialization:
    """Test that JobData can be serialized for caching."""

    def test_jobdata_to_dict(self):
        """Test JobData.to_dict() method."""
        job = JobData(
            job_id="test-123",
            analysis_type="qc",
            file_path="/tmp/test.edf",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            options={"threshold": 0.8},
            progress=0.5,
        )

        # Convert to dict
        job_dict = job.to_dict()

        # Verify it's a dict and has expected fields
        assert isinstance(job_dict, dict)
        assert job_dict["job_id"] == "test-123"
        assert job_dict["analysis_type"] == "qc"
        assert job_dict["status"] == "pending"
        assert job_dict["priority"] == "normal"
        assert job_dict["options"] == {"threshold": 0.8}
        assert job_dict["progress"] == 0.5

        # Verify it's JSON serializable
        json_str = json.dumps(job_dict)
        assert isinstance(json_str, str)

    def test_jobdata_from_dict(self):
        """Test JobData.from_dict() method."""
        job_dict = {
            "job_id": "test-456",
            "analysis_type": "sleep",
            "file_path": "/tmp/test2.edf",
            "status": "processing",
            "priority": "high",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "options": {"window_size": 30},
            "progress": 0.75,
        }

        # Create from dict
        job = JobData.from_dict(job_dict)

        # Verify fields
        assert isinstance(job, JobData)
        assert job.job_id == "test-456"
        assert job.analysis_type == "sleep"
        assert job.status == JobStatus.PROCESSING
        assert job.priority == JobPriority.HIGH
        assert job.options == {"window_size": 30}
        assert job.progress == 0.75

    def test_jobdata_roundtrip(self):
        """Test JobData can be converted to dict and back."""
        original = JobData(
            job_id="test-789",
            analysis_type="events",
            file_path="/tmp/test3.edf",
            status=JobStatus.COMPLETED,
            priority=JobPriority.LOW,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            options={"sensitivity": "high"},
            progress=1.0,
            result={"events_found": 5},
        )

        # Convert to dict and back
        job_dict = original.to_dict()
        reconstructed = JobData.from_dict(job_dict)

        # Verify they match
        assert reconstructed.job_id == original.job_id
        assert reconstructed.analysis_type == original.analysis_type
        assert reconstructed.status == original.status
        assert reconstructed.priority == original.priority
        assert reconstructed.options == original.options
        assert reconstructed.progress == original.progress
        assert reconstructed.result == original.result

    def test_cache_layer_handles_jobdata(self):
        """Test that our cache layer modification works."""
        import dataclasses

        job = JobData(
            job_id="cache-test",
            analysis_type="qc",
            file_path="/tmp/cache.edf",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        # Verify dataclass detection works
        assert dataclasses.is_dataclass(job)
        assert hasattr(job, "to_dict")

        # Create the serialized format our cache uses
        serialized = {"_dataclass_type": job.__class__.__name__, "data": job.to_dict()}

        # Verify it's JSON serializable
        json_str = json.dumps(serialized)
        decoded = json.loads(json_str)

        # Verify we can reconstruct
        assert decoded["_dataclass_type"] == "JobData"
        reconstructed = JobData.from_dict(decoded["data"])
        assert reconstructed.job_id == job.job_id
