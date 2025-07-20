"""Data models for Brain-Go-Brrr API."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from brain_go_brrr.api.schemas import JobPriority, JobStatus


@dataclass(frozen=True)
class Job:
    """Immutable job data model with validation."""

    job_id: str
    analysis_type: str
    file_path: str
    status: JobStatus
    priority: JobPriority
    created_at: datetime
    updated_at: datetime

    # Optional fields
    options: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate job data after initialization."""
        if not self.job_id:
            raise ValueError("job_id cannot be empty")
        if not self.analysis_type:
            raise ValueError("analysis_type cannot be empty")
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        if not isinstance(self.status, JobStatus):
            raise ValueError(f"Invalid status: {self.status}")
        if not isinstance(self.priority, JobPriority):
            raise ValueError(f"Invalid priority: {self.priority}")
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"Progress must be between 0.0 and 1.0, got {self.progress}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "job_id": self.job_id,
            "analysis_type": self.analysis_type,
            "file_path": self.file_path,
            "options": self.options,
            "status": self.status,
            "priority": self.priority,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Create from dictionary."""
        return cls(
            job_id=data["job_id"],
            analysis_type=data["analysis_type"],
            file_path=data["file_path"],
            options=data.get("options", {}),
            status=data["status"]
            if isinstance(data["status"], JobStatus)
            else JobStatus(data["status"]),
            priority=data["priority"]
            if isinstance(data["priority"], JobPriority)
            else JobPriority(data["priority"]),
            progress=data.get("progress", 0.0),
            result=data.get("result"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data["created_at"], str)
            else data["created_at"],
            updated_at=datetime.fromisoformat(data["updated_at"])
            if isinstance(data["updated_at"], str)
            else data["updated_at"],
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at") and isinstance(data["started_at"], str)
            else data.get("started_at"),
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at") and isinstance(data["completed_at"], str)
            else data.get("completed_at"),
        )
