"""Unit tests for ThreadSafeJobStore (behavioral, no mocks)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from brain_go_brrr.application.jobs.models import JobData, JobPriority, JobStatus
from brain_go_brrr.application.jobs.store import ThreadSafeJobStore, get_job_store


def _job(id_: str, status: JobStatus = JobStatus.PENDING) -> JobData:
    return JobData(
        id=id_,
        type="abnormality",
        status=status,
        priority=JobPriority.NORMAL,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        started_at=None,
        completed_at=None,
        error=None,
        result=None,
        metadata={},
    )


class TestThreadSafeJobStore:
    def test_create_get_update_patch_delete(self) -> None:
        store = ThreadSafeJobStore()

        # Create
        j = _job("a1")
        store.create(j.id, j)
        assert store.get("a1").id == "a1"

        # Update fields via update
        assert store.update("a1", {"status": JobStatus.PROCESSING})
        assert store.get("a1").status == JobStatus.PROCESSING

        # Patch valid field
        assert store.patch("a1", status=JobStatus.COMPLETED)
        assert store.get("a1").status == JobStatus.COMPLETED

        # Invalid patch fields raise
        with pytest.raises(ValueError):
            store.patch("a1", not_a_field=123)  # type: ignore[arg-type]

        # Delete
        assert store.delete("a1") is True
        assert store.get("a1") is None

    def test_list_and_counts_and_cleanup(self) -> None:
        store = ThreadSafeJobStore()
        # Add mixed statuses with timestamps
        now = datetime.now(timezone.utc)
        for i in range(5):
            j = _job(f"c{i}", status=JobStatus.COMPLETED)
            j = JobData.from_dict({**j.to_dict(), "updated_at": now - timedelta(minutes=i)})
            store.create(j.id, j)
        for i in range(3):
            j = _job(f"f{i}", status=JobStatus.FAILED)
            j = JobData.from_dict({**j.to_dict(), "updated_at": now - timedelta(minutes=i)})
            store.create(j.id, j)

        lst = store.list_by_status(JobStatus.COMPLETED)
        assert len(lst) == 5
        counts = store.count_by_status()
        assert counts["completed"] == 5
        assert counts["failed"] == 3

        # Keep only 2 most recent of each category
        deleted = store.cleanup_old_jobs(keep_completed=2, keep_failed=2)
        assert deleted == (5 - 2) + (3 - 2)
        counts2 = store.count_by_status()
        assert counts2["completed"] == 2
        assert counts2["failed"] == 2

    def test_global_store_singleton_like(self) -> None:
        s1 = get_job_store()
        s2 = get_job_store()
        assert s1 is s2
