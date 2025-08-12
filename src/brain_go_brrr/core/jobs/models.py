"""Domain models for job management.

This module contains the core job-related entities that should not depend
on any infrastructure or API layer (following SOLID's Dependency Inversion).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class JobData:
    """Core job data model.
    
    This is the domain entity for jobs. The API layer should map
    from this to its own DTOs/schemas for external communication.
    """

    id: str
    type: str
    status: JobStatus
    priority: JobPriority = JobPriority.NORMAL
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: Any = None
    metadata: dict[str, Any] | None = None


__all__ = ["JobData", "JobPriority", "JobStatus"]
