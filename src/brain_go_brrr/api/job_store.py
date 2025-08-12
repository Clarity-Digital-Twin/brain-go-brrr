"""API-layer job store that uses API JobData models."""

import logging
import threading
from typing import Any

from brain_go_brrr.api.schemas import JobData, JobStatus

logger = logging.getLogger(__name__)


class APIJobStore:
    """Thread-safe job store for API layer.

    This store uses the API JobData model with job_id and analysis_type fields,
    providing compatibility with the existing API endpoints.
    """

    def __init__(self) -> None:
        """Initialize the job store with thread safety."""
        self._jobs: dict[str, JobData] = {}
        self._lock = threading.RLock()
        logger.info("Initialized API job store")

    def create(self, job_id: str, job_data: JobData) -> None:
        """Create a new job entry.

        Args:
            job_id: Unique job identifier
            job_data: Job data object

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

            # Create new immutable JobData
            new_job = JobData.from_dict(job_dict)
            self._jobs[job_id] = new_job
            logger.debug(f"Updated job {job_id}: {list(updates.keys())}")
            return True

    def patch(self, job_id: str, **fields: Any) -> bool:
        """Patch specific job fields.

        Args:
            job_id: Job identifier
            **fields: Specific fields to update

        Returns:
            True if patched, False if job not found
        """
        return self.update(job_id, fields)

    def list_all(self) -> list[JobData]:
        """List all jobs.

        Returns:
            List of all job data objects
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

    def clear(self) -> None:
        """Clear all jobs."""
        with self._lock:
            self._jobs.clear()
            logger.info("Cleared all jobs from store")


# Global instance for API layer
_api_job_store: APIJobStore | None = None
_store_lock = threading.Lock()


def get_api_job_store() -> APIJobStore:
    """Get or create the global API job store instance.

    Returns:
        The singleton APIJobStore instance
    """
    global _api_job_store
    with _store_lock:
        if _api_job_store is None:
            _api_job_store = APIJobStore()
        return _api_job_store
