"""Thread-safe job store implementation for managing analysis jobs."""

import logging
import threading
from datetime import datetime
from typing import Any

from brain_go_brrr.api.schemas import JobData, JobPriority, JobStatus

logger = logging.getLogger(__name__)


class ThreadSafeJobStore:
    """Thread-safe in-memory job store.

    This is a simple implementation suitable for development and single-instance
    deployments. For production with multiple workers, use Redis or a database.
    """

    def __init__(self) -> None:
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
        """Update job fields by creating a new immutable JobData.

        Args:
            job_id: Job identifier
            updates: Fields to update

        Returns:
            True if updated, False if job not found
        """
        with self._lock:
            if job_id not in self._jobs:
                return False

            # Get current job and convert to dict
            current_job = self._jobs[job_id]
            job_dict = current_job.to_dict()

            # Apply updates
            job_dict.update(updates)

            # Handle datetime updates
            if "updated_at" in updates and isinstance(updates["updated_at"], str):
                job_dict["updated_at"] = updates["updated_at"]

            # Create new immutable JobData
            if hasattr(JobData, "from_dict"):
                new_job = JobData.from_dict(job_dict)
            else:
                # This shouldn't happen with dataclass, but kept for safety
                raise RuntimeError("JobData.from_dict method not found")

            self._jobs[job_id] = new_job
            logger.debug(f"Updated job {job_id}: {list(updates.keys())}")
            return True

    def patch(self, job_id: str, **fields: Any) -> bool:
        """Patch specific job fields without replacing the entire record.

        Args:
            job_id: Job identifier
            **fields: Specific fields to update

        Returns:
            True if patched, False if job not found

        Raises:
            ValueError: If trying to patch non-existent fields
        """
        with self._lock:
            if job_id not in self._jobs:
                return False

            # Get current job
            current_job = self._jobs[job_id]

            # Get valid field names from JobData class
            valid_fields = {
                "job_id",
                "analysis_type",
                "file_path",
                "status",
                "priority",
                "created_at",
                "updated_at",
                "options",
                "progress",
                "result",
                "error",
                "started_at",
                "completed_at",
            }

            # Validate fields exist in JobData schema
            invalid_fields = [k for k in fields if k not in valid_fields]
            if invalid_fields:
                raise ValueError(f"Cannot patch non-existent fields: {invalid_fields}")

            # Convert current job to dict
            job_dict = current_job.to_dict()

            # Apply patches to the dict
            for key, value in fields.items():
                # Handle special cases for enums
                if (key == "status" and isinstance(value, JobStatus)) or (
                    key == "priority" and isinstance(value, JobPriority)
                ):
                    job_dict[key] = value.value
                # Handle datetime objects
                elif key in ("created_at", "updated_at", "started_at", "completed_at"):
                    if isinstance(value, datetime):
                        job_dict[key] = value.isoformat()
                    else:
                        job_dict[key] = value
                else:
                    job_dict[key] = value

            # Create new immutable JobData instance
            new_job = JobData.from_dict(job_dict)

            # Replace the old instance in the store
            self._jobs[job_id] = new_job

            logger.debug(f"Patched job {job_id}: {list(fields.keys())}")
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
            return [job for job in self._jobs.values() if job.status == status]

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
                status = job.status.value
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
            sorted_jobs = sorted(self._jobs.items(), key=lambda x: x[1].updated_at, reverse=True)

            completed_count = 0
            failed_count = 0
            to_delete = []

            for job_id, job in sorted_jobs:
                status = job.status

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
