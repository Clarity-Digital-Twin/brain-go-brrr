"""Tests for core.exceptions module - CLEAN, NO BULLSHIT."""

import pytest
from src.brain_go_brrr.core.exceptions import (
    BrainGoBrrrError,
    ConfigurationError,
    EdfLoadError,
    FileFormatError,
    InsufficientDataError,
    ProcessingError,
    QualityCheckError,
    SleepAnalysisError,
    UnsupportedMontageError,
    FeatureExtractionError,
    AbnormalityDetectionError,
    ModelError,
    ModelNotInitializedError,
    ModelLoadError,
    ModelInferenceError,
    ResourceError,
    InsufficientMemoryError,
    GPUNotAvailableError,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""
    
    def test_base_exception(self):
        """Test base exception class."""
        exc = BrainGoBrrrError("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)
    
    def test_configuration_error(self):
        """Test configuration error."""
        exc = ConfigurationError("Bad config")
        assert str(exc) == "Bad config"
        assert isinstance(exc, BrainGoBrrrError)
    
    def test_processing_error(self):
        """Test processing error."""
        exc = ProcessingError("Cannot process data")
        assert str(exc) == "Cannot process data"
        assert isinstance(exc, BrainGoBrrrError)
    
    def test_edf_load_error(self):
        """Test EDF-specific load error."""
        exc = EdfLoadError("Invalid EDF file")
        assert str(exc) == "Invalid EDF file"
        assert isinstance(exc, ProcessingError)
        assert isinstance(exc, BrainGoBrrrError)
    
    def test_sleep_analysis_error(self):
        """Test sleep analysis error."""
        exc = SleepAnalysisError("Analysis failed")
        assert str(exc) == "Analysis failed"
        assert isinstance(exc, BrainGoBrrrError)
    
    def test_unsupported_montage_error(self):
        """Test unsupported montage error."""
        exc = UnsupportedMontageError("Unknown montage")
        assert str(exc) == "Unknown montage"
        assert isinstance(exc, SleepAnalysisError)
    
    def test_model_load_error(self):
        """Test model load error."""
        exc = ModelLoadError("Cannot load model")
        assert str(exc) == "Cannot load model"
        assert isinstance(exc, BrainGoBrrrError)
    
    def test_feature_extraction_error(self):
        """Test feature extraction error."""
        exc = FeatureExtractionError("Feature extraction failed")
        assert str(exc) == "Feature extraction failed"
        assert isinstance(exc, BrainGoBrrrError)
    
    def test_model_inference_error(self):
        """Test model inference error."""
        exc = ModelInferenceError("Inference failed")
        assert str(exc) == "Inference failed"
        assert isinstance(exc, ModelError)
        assert isinstance(exc, BrainGoBrrrError)
    
    def test_resource_errors(self):
        """Test resource-related errors."""
        exc1 = InsufficientMemoryError("Out of memory")
        assert isinstance(exc1, ResourceError)
        
        exc2 = GPUNotAvailableError("No GPU")
        assert isinstance(exc2, ResourceError)


class TestExceptionUsage:
    """Test exception usage patterns."""
    
    def test_exception_chaining(self):
        """Test exception chaining preserves context."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise EdfLoadError("EDF failed") from e
        except EdfLoadError as e:
            assert str(e) == "EDF failed"
            assert e.__cause__ is not None
            assert str(e.__cause__) == "Original error"
    
    def test_exception_with_details(self):
        """Test exceptions can carry additional details."""
        exc = ConfigurationError("Missing required field: sample_rate")
        assert "sample_rate" in str(exc)
    
    def test_catching_hierarchy(self):
        """Test catching exceptions by base class."""
        errors = [
            EdfLoadError("EDF"),
            ConfigurationError("Config"),
            ModelLoadError("Model"),
        ]
        
        for error in errors:
            with pytest.raises(BrainGoBrrrError):
                raise error
    
    def test_specific_catch(self):
        """Test catching specific exceptions."""
        with pytest.raises(EdfLoadError):
            raise EdfLoadError("Bad EDF")
        
        # Should not catch different error
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Bad config")