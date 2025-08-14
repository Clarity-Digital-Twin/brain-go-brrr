"""FastAPI dependency injection for Clean Architecture.

This module provides dependency injection for API endpoints,
ensuring clean separation of concerns and testability.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from brain_go_brrr.application.factories_pure import (
    create_feature_extractor_pure,
    create_pure_abnormality_detector,
    create_quality_controller_pure,
)
from brain_go_brrr.application.factories_types import (
    AbnormalityDetectorPort,
    FeatureExtractorPort,
    QualityControllerPort,
)
from brain_go_brrr.core.config import get_settings


@lru_cache(maxsize=1)
def get_quality_controller() -> QualityControllerPort:
    """Get or create QC controller with caching.

    This is the ONLY place where the QC controller is instantiated.
    No global initialization, no file touching, pure DI.
    """
    settings = get_settings()

    # Get model path from settings or use default
    model_path = getattr(settings, "eegpt_model_path", None)
    if not model_path:
        model_path = "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    return create_quality_controller_pure(
        model_path=str(model_path),
        device="auto",
        enable_autoreject=True,
    )


@lru_cache(maxsize=1)
def get_abnormality_detector() -> AbnormalityDetectorPort:
    """Get or create abnormality detector with caching.

    Singleton pattern through lru_cache ensures we don't
    reload models unnecessarily.
    """
    settings = get_settings()

    model_path = getattr(settings, "eegpt_model_path", None)
    if not model_path:
        model_path = "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    return create_pure_abnormality_detector(
        model_path=str(model_path),
        device="auto",
    )


@lru_cache(maxsize=1)
def get_feature_extractor() -> FeatureExtractorPort:
    """Get or create feature extractor with caching."""
    settings = get_settings()

    model_path = getattr(settings, "eegpt_model_path", None)
    if not model_path:
        model_path = "data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"

    return create_feature_extractor_pure(
        model_path=str(model_path),
        device="auto",
        window_size=4.0,
        overlap=0.5,
    )


# Type aliases for cleaner endpoint signatures
QCController = Annotated[QualityControllerPort, Depends(get_quality_controller)]
AbnormalityDetector = Annotated[AbnormalityDetectorPort, Depends(get_abnormality_detector)]
FeatureExtractor = Annotated[FeatureExtractorPort, Depends(get_feature_extractor)]
