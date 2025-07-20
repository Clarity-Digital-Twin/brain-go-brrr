"""Central route configuration for Brain-Go-Brrr API."""

from dataclasses import dataclass


@dataclass(frozen=True)
class APIRoutes:
    """Central configuration for all API routes."""

    # Base prefixes
    API_V1 = "/api/v1"

    # Health endpoints
    HEALTH = f"{API_V1}/health"
    READY = f"{API_V1}/ready"

    # Job management
    JOBS_CREATE = f"{API_V1}/jobs/create"
    JOBS_STATUS = f"{API_V1}/jobs/{{job_id}}/status"
    JOBS_LIST = f"{API_V1}/jobs"

    # EEG analysis
    QC_ANALYZE = f"{API_V1}/eeg/analyze"
    SLEEP_ANALYZE = f"{API_V1}/eeg/sleep/analyze"

    # Queue management
    QUEUE_STATUS = f"{API_V1}/queue/status"

    # Cache operations
    CACHE_STATUS = f"{API_V1}/cache/status"

    # Documentation
    DOCS = "/api/docs"
    REDOC = "/api/redoc"


# Singleton instance
routes = APIRoutes()
