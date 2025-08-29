"""Behavioral tests for ThreadSafeJobStore - NO MOCKING, REAL THREAD SAFETY."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from brain_go_brrr.application.jobs.models import JobData, JobPriority, JobStatus
from brain_go_brrr.application.jobs.store import ThreadSafeJobStore


class TestJobStoreBasicOperations:
    """Test basic CRUD operations of job store."""

    def test_create_and_get_job(self):
        """Test creating and retrieving a job."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="test-123",
            analysis_type="sleep",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
        )

        # Create job
        store.create("test-123", job_data)

        # Retrieve job
        retrieved = store.get("test-123")
        assert retrieved is not None
        assert retrieved.job_id == "test-123"
        assert retrieved.analysis_type == "sleep"
        assert retrieved.status == JobStatus.PENDING

    def test_create_duplicate_raises(self):
        """Test creating duplicate job raises ValueError."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="dup-123",
            analysis_type="qc",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
        )

        # First create succeeds
        store.create("dup-123", job_data)

        # Second create fails
        with pytest.raises(ValueError, match="Job dup-123 already exists"):
            store.create("dup-123", job_data)

    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent job returns None."""
        store = ThreadSafeJobStore()
        result = store.get("does-not-exist")
        assert result is None

    def test_update_job_fields(self):
        """Test updating job fields."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="update-123",
            analysis_type="abnormal",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
        )

        store.create("update-123", job_data)

        # Update status and add result
        success = store.update(
            "update-123",
            {
                "status": JobStatus.COMPLETED,
                "result": {"abnormal": False, "confidence": 0.95},
            },
        )

        assert success is True

        # Verify updates
        updated = store.get("update-123")
        assert updated is not None
        assert updated.status == JobStatus.COMPLETED
        assert updated.result == {"abnormal": False, "confidence": 0.95}

    def test_update_nonexistent_returns_false(self):
        """Test updating nonexistent job returns False."""
        store = ThreadSafeJobStore()
        success = store.update("nonexistent", {"status": JobStatus.FAILED})
        assert success is False

    def test_patch_specific_fields(self):
        """Test patching specific job fields."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="patch-123",
            analysis_type="sleep",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
        )

        store.create("patch-123", job_data)

        # Patch priority and progress
        success = store.patch("patch-123", priority=JobPriority.HIGH, progress=50.0)
        assert success is True

        # Verify patches
        patched = store.get("patch-123")
        assert patched is not None
        assert patched.priority == JobPriority.HIGH
        assert patched.progress == 50.0
        # Other fields unchanged
        assert patched.status == JobStatus.PENDING
        assert patched.analysis_type == "sleep"

    def test_patch_invalid_field_raises(self):
        """Test patching invalid field raises ValueError."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="invalid-patch",
            analysis_type="qc",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
        )

        store.create("invalid-patch", job_data)

        # Try to patch non-existent field
        with pytest.raises(ValueError, match="Cannot patch non-existent fields"):
            store.patch("invalid-patch", fake_field="value")

    def test_patch_nonexistent_job_returns_false(self):
        """Test patching nonexistent job returns False."""
        store = ThreadSafeJobStore()
        success = store.patch("nonexistent", status=JobStatus.FAILED)
        assert success is False


class TestJobStoreThreadSafety:
    """Test REAL thread safety behavior."""

    def test_concurrent_creates_different_jobs(self):
        """Test concurrent creation of different jobs works correctly."""
        store = ThreadSafeJobStore()
        results = []
        errors = []

        def create_job(job_id: str):
            try:
                job_data = JobData(
                    job_id=job_id,
                    analysis_type="test",
                    file_path=f"/data/{job_id}.edf",
                    status=JobStatus.PENDING,
                )
                store.create(job_id, job_data)
                results.append(job_id)
            except Exception as e:
                errors.append(str(e))

        # Create 100 jobs concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_job, f"job-{i}") for i in range(100)]
            for future in futures:
                future.result()

        assert len(results) == 100
        assert len(errors) == 0

        # Verify all jobs exist
        for i in range(100):
            job = store.get(f"job-{i}")
            assert job is not None
            assert job.job_id == f"job-{i}"

    def test_concurrent_creates_same_job(self):
        """Test concurrent creation of same job - only one should succeed."""
        store = ThreadSafeJobStore()
        successes = []
        failures = []

        def try_create():
            try:
                job_data = JobData(
                    job_id="same-job",
                    analysis_type="test",
                    file_path="/data/test.edf",
                    status=JobStatus.PENDING,
                )
                store.create("same-job", job_data)
                successes.append(1)
            except ValueError:
                failures.append(1)

        # Try to create same job 50 times concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_create) for _ in range(50)]
            for future in futures:
                future.result()

        # Exactly one should succeed
        assert len(successes) == 1
        assert len(failures) == 49

    def test_concurrent_updates(self):
        """Test concurrent updates to same job maintain consistency."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="update-race",
            analysis_type="test",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
            progress=0.0,
        )
        store.create("update-race", job_data)

        def increment_progress(value: float):
            current = store.get("update-race")
            if current:
                # Simulate read-modify-write race condition
                time.sleep(0.001)  # Small delay to increase chance of race
                store.update("update-race", {"progress": current.progress + value})

        # 100 threads each incrementing by 1.0
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(increment_progress, 1.0) for _ in range(100)]
            for future in futures:
                future.result()

        # Due to race conditions, final value might not be exactly 100
        # But it should be > 0 and <= 100 (no corruption)
        final = store.get("update-race")
        assert final is not None
        assert 0 < final.progress <= 100

    def test_concurrent_read_while_updating(self):
        """Test reading doesn't block or get corrupted during updates."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="read-update",
            analysis_type="test",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
        )
        store.create("read-update", job_data)

        read_values = []
        update_count = [0]

        def reader():
            for _ in range(100):
                job = store.get("read-update")
                if job:
                    read_values.append(job.status)
                time.sleep(0.001)

        def updater():
            statuses = [JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.PENDING]
            for i in range(30):
                store.update("read-update", {"status": statuses[i % 3]})
                update_count[0] += 1
                time.sleep(0.003)

        # Start reader and updater threads
        reader_thread = threading.Thread(target=reader)
        updater_thread = threading.Thread(target=updater)

        reader_thread.start()
        updater_thread.start()

        reader_thread.join()
        updater_thread.join()

        # Should have read values and all should be valid JobStatus
        assert len(read_values) > 0
        assert all(isinstance(v, JobStatus) for v in read_values)
        assert update_count[0] == 30

    def test_delete_job(self):
        """Test delete operation."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="delete-me",
            analysis_type="test",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
        )

        store.create("delete-me", job_data)
        assert store.get("delete-me") is not None

        # Delete job
        success = store.delete("delete-me")
        assert success is True
        assert store.get("delete-me") is None

        # Delete again should return False
        success = store.delete("delete-me")
        assert success is False

    def test_list_jobs(self):
        """Test listing all jobs."""
        store = ThreadSafeJobStore()

        # Create several jobs
        for i in range(5):
            job_data = JobData(
                job_id=f"list-{i}",
                analysis_type="test",
                file_path=f"/data/test-{i}.edf",
                status=JobStatus.PENDING if i % 2 == 0 else JobStatus.COMPLETED,
            )
            store.create(f"list-{i}", job_data)

        # List all jobs
        all_jobs = store.list_jobs()
        assert len(all_jobs) == 5
        assert all(job.job_id.startswith("list-") for job in all_jobs)

    def test_list_jobs_by_status(self):
        """Test filtering jobs by status."""
        store = ThreadSafeJobStore()

        # Create mix of statuses
        statuses = [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED]
        for i, status in enumerate(statuses * 2):  # 8 jobs total
            job_data = JobData(
                job_id=f"status-{i}",
                analysis_type="test",
                file_path=f"/data/test-{i}.edf",
                status=status,
            )
            store.create(f"status-{i}", job_data)

        # Filter by status
        pending = store.list_jobs_by_status(JobStatus.PENDING)
        assert len(pending) == 2
        assert all(job.status == JobStatus.PENDING for job in pending)

        completed = store.list_jobs_by_status(JobStatus.COMPLETED)
        assert len(completed) == 2
        assert all(job.status == JobStatus.COMPLETED for job in completed)

    def test_update_with_datetime(self):
        """Test updating with datetime fields."""
        store = ThreadSafeJobStore()
        job_data = JobData(
            job_id="datetime-test",
            analysis_type="test",
            file_path="/data/test.edf",
            status=JobStatus.PENDING,
        )

        store.create("datetime-test", job_data)

        # Update with datetime
        now = datetime.now()
        success = store.update(
            "datetime-test",
            {
                "status": JobStatus.RUNNING,
                "started_at": now.isoformat(),
            },
        )

        assert success is True
        updated = store.get("datetime-test")
        assert updated is not None
        assert updated.status == JobStatus.RUNNING
        assert updated.started_at == now.isoformat()