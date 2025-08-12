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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobData":
        """Create JobData from dictionary (for deserialization)."""
        return cls(
            id=data["id"],
            type=data["type"],
            status=JobStatus(data["status"]),
            priority=JobPriority(data.get("priority", JobPriority.NORMAL)),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            result=data.get("result"),
            metadata=data.get("metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert JobData to dictionary (for serialization)."""
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "result": self.result,
            "metadata": self.metadata,
        }


__all__ = ["JobData", "JobPriority", "JobStatus"]
