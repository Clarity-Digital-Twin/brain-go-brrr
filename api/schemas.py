"""Pydantic models for Brain-Go-Brrr API."""

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


class JobData(TypedDict):
    """Job data structure for internal storage."""

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
