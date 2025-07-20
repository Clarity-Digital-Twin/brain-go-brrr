"""Pydantic models for Brain-Go-Brrr API."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    """Job priority levels."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass(frozen=True)
class JobData:
    """Immutable job data structure for internal storage.

    This is a frozen dataclass to prevent accidental mutations.
    Use JobStore.update() or JobStore.patch() for modifications.
    """

    job_id: str
    analysis_type: str
    file_path: str
    status: JobStatus
    priority: JobPriority
    created_at: datetime
    updated_at: datetime

    # Optional fields with defaults
    options: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self):
        """Validate data after initialization."""
        if not self.job_id:
            raise ValueError("job_id cannot be empty")
        if not self.analysis_type:
            raise ValueError("analysis_type cannot be empty")
        if not self.file_path:
            raise ValueError("file_path cannot be empty")
        # These checks are not needed with frozen dataclass
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError(f"Progress must be between 0.0 and 1.0, got {self.progress}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "job_id": self.job_id,
            "analysis_type": self.analysis_type,
            "file_path": self.file_path,
            "options": self.options,
            "status": self.status.value,
            "priority": self.priority.value,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobData":
        """Create from dictionary (e.g., from storage)."""
        # Convert string timestamps back to datetime
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        started_at = data.get("started_at")
        if started_at and isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)

        completed_at = data.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        return cls(
            job_id=data["job_id"],
            analysis_type=data["analysis_type"],
            file_path=data["file_path"],
            options=data.get("options", {}),
            status=JobStatus(data["status"]) if isinstance(data["status"], str) else data["status"],
            priority=JobPriority(data["priority"])
            if isinstance(data["priority"], str)
            else data["priority"],
            progress=data.get("progress", 0.0),
            result=data.get("result"),
            error=data.get("error"),
            created_at=created_at if created_at else datetime.utcnow(),
            updated_at=updated_at if updated_at else datetime.utcnow(),
            started_at=started_at,
            completed_at=completed_at,
        )


# Keep TypedDict for backward compatibility during migration
class JobDataDict(TypedDict):
    """Legacy job data structure - use JobData dataclass instead."""

    job_id: str
    analysis_type: str
    file_path: str
    options: dict[str, Any]
    status: JobStatus
    priority: JobPriority
    progress: float | None
    result: dict[str, Any] | None
    error: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    completed_at: str | None


class AnalysisRequest(BaseModel):
    """Request model for EEG analysis."""

    file_path: str
    analysis_type: str = "comprehensive"


class QCResponse(BaseModel):
    """Response model for quality control checks."""

    flag: str
    confidence: float
    bad_channels: list[str]
    quality_metrics: dict[str, Any]
    recommendation: str
    processing_time: float
    quality_grade: str
    timestamp: str
    error: str | None = None
    cached: bool = False


class SleepAnalysisResponse(BaseModel):
    """Response model for sleep analysis."""

    status: str
    sleep_stages: dict[str, float]
    sleep_metrics: dict[str, float]
    hypnogram: list[dict[str, Any]]
    metadata: dict[str, Any]
    processing_time: float
    timestamp: str
    error: str | None = None
    cached: bool = False


class JobCreateRequest(BaseModel):
    """Request model for creating a new job."""

    analysis_type: str = Field(..., description="Type of analysis to perform")
    file_path: str = Field(..., description="Path to the file to analyze")
    options: dict[str, Any] = Field(default_factory=dict, description="Analysis options")
    priority: JobPriority = Field(default=JobPriority.NORMAL, description="Job priority")


class JobResponse(BaseModel):
    """Response model for job information."""

    job_id: str
    analysis_type: str
    status: JobStatus
    priority: JobPriority
    progress: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None


class JobListResponse(BaseModel):
    """Response model for job list."""

    jobs: list[JobResponse]
    total: int
    page: int = 1
    page_size: int = 100
    has_next: bool = False


class QueueStatusResponse(BaseModel):
    """Response model for queue status."""

    pending_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    workers_active: int
    queue_health: str


class CacheWarmupRequest(BaseModel):
    """Request model for cache warmup."""

    file_content: str  # Base64 encoded
    analysis_types: list[str] = ["basic", "detailed"]
