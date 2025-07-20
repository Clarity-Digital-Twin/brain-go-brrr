"""Thread-safe job store implementation for managing analysis jobs."""

import logging
import threading
from typing import Any

from api.routers.jobs import JobData
from api.schemas import JobStatus

logger = logging.getLogger(__name__)


class ThreadSafeJobStore:
    """Thread-safe in-memory job store.

    This is a simple implementation suitable for development and single-instance
    deployments. For production with multiple workers, use Redis or a database.
    """

    def __init__(self):
        """Initialize the job store with thread safety."""
        self._jobs: dict[str, JobData] = {}
        self._lock = threading.RLock()  # Reentrant lock for nested access
        logger.info("Initialized thread-safe job store")

    def create(self, job_id: str, job_data: JobData) -> None:
        """Create a new job entry.

        Args:
            job_id: Unique job identifier
            job_data: Job data dictionary

        Raises:
            ValueError: If job_id already exists
        """
        with self._lock:
            if job_id in self._jobs:
                raise ValueError(f"Job {job_id} already exists")
            self._jobs[job_id] = job_data
            logger.debug(f"Created job {job_id}")

    def get(self, job_id: str) -> JobData | None:
        """Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job data if found, None otherwise
        """
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, updates: dict[str, Any]) -> bool:
        """Update job fields.

        Args:
            job_id: Job identifier
            updates: Fields to update

        Returns:
            True if updated, False if job not found
        """
        with self._lock:
            if job_id not in self._jobs:
                return False

            # Type-safe update
            job = self._jobs[job_id]
            for key, value in updates.items():
                if key in job:
                    job[key] = value  # type: ignore[literal-required]

            logger.debug(f"Updated job {job_id}: {list(updates.keys())}")
            return True

    def delete(self, job_id: str) -> bool:
        """Delete a job.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.debug(f"Deleted job {job_id}")
                return True
            return False

    def list_all(self) -> list[JobData]:
        """List all jobs.

        Returns:
            List of all job data
        """
        with self._lock:
            return list(self._jobs.values())

    def list_by_status(self, status: JobStatus) -> list[JobData]:
        """List jobs by status.

        Args:
            status: Job status to filter by

        Returns:
            List of jobs with matching status
        """
        with self._lock:
            return [job for job in self._jobs.values() if job["status"] == status]

    def count_by_status(self) -> dict[str, int]:
        """Count jobs by status.

        Returns:
            Dictionary of status counts
        """
        with self._lock:
            counts = {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
            }

            for job in self._jobs.values():
                status = job["status"].value
                if status in counts:
                    counts[status] += 1

            return counts

    def cleanup_old_jobs(self, keep_completed: int = 100, keep_failed: int = 50) -> int:
        """Clean up old completed and failed jobs.

        Args:
            keep_completed: Number of recent completed jobs to keep
            keep_failed: Number of recent failed jobs to keep

        Returns:
            Number of jobs deleted
        """
        with self._lock:
            # Sort jobs by updated_at timestamp
            sorted_jobs = sorted(
                self._jobs.items(), key=lambda x: x[1].get("updated_at", ""), reverse=True
            )

            completed_count = 0
            failed_count = 0
            to_delete = []

            for job_id, job in sorted_jobs:
                status = job["status"]

                if status == JobStatus.COMPLETED:
                    completed_count += 1
                    if completed_count > keep_completed:
                        to_delete.append(job_id)

                elif status == JobStatus.FAILED:
                    failed_count += 1
                    if failed_count > keep_failed:
                        to_delete.append(job_id)

            # Delete old jobs
            for job_id in to_delete:
                del self._jobs[job_id]

            if to_delete:
                logger.info(f"Cleaned up {len(to_delete)} old jobs")

            return len(to_delete)

    def clear(self) -> None:
        """Clear all jobs (use with caution)."""
        with self._lock:
            count = len(self._jobs)
            self._jobs.clear()
            logger.warning(f"Cleared all {count} jobs from store")


# Global instance
_job_store: ThreadSafeJobStore | None = None
_store_lock = threading.Lock()


def get_job_store() -> ThreadSafeJobStore:
    """Get or create the global job store instance.

    Returns:
        Thread-safe job store instance
    """
    global _job_store

    if _job_store is None:
        with _store_lock:
            # Double-check pattern
            if _job_store is None:
                _job_store = ThreadSafeJobStore()

    return _job_store
