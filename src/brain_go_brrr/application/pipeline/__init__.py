"""Processing pipeline core module."""

from .parallel import ParallelEEGPipeline
from .eegpt_orchestration import predict_abnormality_with_eegpt

__all__ = ["ParallelEEGPipeline", "predict_abnormality_with_eegpt"]
