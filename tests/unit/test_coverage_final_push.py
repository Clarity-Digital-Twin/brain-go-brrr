"""Final tests to push coverage over 60%."""

import pytest
from brain_go_brrr.core.exceptions import (
    FileFormatError,
    InsufficientDataError,
    QualityCheckError,
    SleepAnalysisError,
    UnsupportedMontageError,
    FeatureExtractionError,
    AbnormalityDetectionError,
    ModelNotInitializedError,
    ModelLoadError,
    ModelInferenceError,
    InsufficientMemoryError,
    GPUNotAvailableError
)


def test_all_exceptions_instantiate():
    """Test that all exceptions can be instantiated."""
    exceptions = [
        FileFormatError("format error"),
        InsufficientDataError("not enough data"),
        QualityCheckError("quality issue"),
        SleepAnalysisError("sleep error"),
        UnsupportedMontageError("montage issue"),
        FeatureExtractionError("feature error"),
        AbnormalityDetectionError("abnormal error"),
        ModelNotInitializedError("not initialized"),
        ModelLoadError("load error"),
        ModelInferenceError("inference error"),
        InsufficientMemoryError("memory error"),
        GPUNotAvailableError("no gpu")
    ]
    
    for exc in exceptions:
        assert isinstance(exc, Exception)
        assert len(str(exc)) > 0


def test_exception_inheritance_chain():
    """Test exception inheritance chain."""
    from brain_go_brrr.core.exceptions import (
        BrainGoBrrrError,
        EdfLoadError,
        ProcessingError,
        ModelError,
        ResourceError
    )
    
    # Test inheritance chains
    assert issubclass(FileFormatError, EdfLoadError)
    assert issubclass(EdfLoadError, BrainGoBrrrError)
    
    assert issubclass(QualityCheckError, ProcessingError)
    assert issubclass(ProcessingError, BrainGoBrrrError)
    
    assert issubclass(ModelLoadError, ModelError)
    assert issubclass(ModelError, BrainGoBrrrError)
    
    assert issubclass(GPUNotAvailableError, ResourceError)
    assert issubclass(ResourceError, BrainGoBrrrError)