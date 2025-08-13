"""Application layer - orchestration and use cases."""

# Re-export commonly used application components
from .jobs.models import JobData, JobStatus
from .pipeline.parallel import ParallelEEGPipeline

# Alias for backward compatibility
ParallelPipeline = ParallelEEGPipeline

__all__ = [
    "JobData",
    "JobStatus",
    "ParallelEEGPipeline",
    "ParallelPipeline",  # Alias
]
