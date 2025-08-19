"""Processing pipeline core module."""

from .eegpt_orchestration import predict_abnormality_with_eegpt
from .parallel import ParallelEEGPipeline

__all__ = ["ParallelEEGPipeline", "predict_abnormality_with_eegpt"]
