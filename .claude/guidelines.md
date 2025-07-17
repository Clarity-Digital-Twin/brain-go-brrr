# AI Agent Development Guidelines
## EEGPT Clinical Decision Support System

### Purpose
This document provides specific guidelines for AI agents (Claude, GitHub Copilot, etc.) working on this codebase. Follow these standards to ensure consistent, high-quality code generation.

---

## Project Context

### What We're Building
- FDA Class II medical device software for EEG analysis
- Clinical decision support system using EEGPT foundation model
- Must comply with HIPAA, 21 CFR Part 11, and medical device regulations
- Lives depend on accuracy - no room for errors

### Key Documents to Read First
1. `/docs/literature-master-reference.md` - Technical specifications from research
2. `/docs/PRD-product-requirements.md` - What we're building and why
3. `/docs/TRD-technical-requirements.md` - How we're building it
4. `/docs/BDD-behavior-specifications.md` - Expected behaviors

---

## Code Generation Standards

### Python Code Style
```python
"""
ALWAYS include comprehensive module docstrings like this example.

Module: module_name.py
Purpose: Clear, one-line purpose statement
Requirements: Link to specific PRD/TRD requirements (e.g., FR1.1, NFR2.3)
Dependencies: List all external dependencies and data paths

AI Agent Instructions:
- Specific guidance for code generation in this module
- Key patterns to follow
- Common pitfalls to avoid
"""

from typing import Protocol, Optional, List, Dict, Any
import logging
from pathlib import Path

# Use Protocol for interfaces (not ABC)
class IQualityAnalyzer(Protocol):
    """Interface for quality analysis implementations."""
    
    async def analyze(self, raw_eeg: Any) -> QualityResult:
        """Analyze EEG quality with specific metrics."""
        ...

# Implement SOLID principles
class AutorejectQualityAnalyzer:
    """Concrete implementation using Autoreject algorithm."""
    
    def __init__(self, 
                 threshold_optimizer: IThresholdOptimizer,
                 logger: logging.Logger):
        # Dependency injection only
        self._optimizer = threshold_optimizer
        self._logger = logger
    
    async def analyze(self, raw_eeg: Any) -> QualityResult:
        """
        Analyze EEG quality using Autoreject algorithm.
        
        Args:
            raw_eeg: MNE Raw object with EEG data
            
        Returns:
            QualityResult with bad channels and metrics
            
        Raises:
            QualityAnalysisError: If analysis fails
        """
        # Always use correlation IDs for tracing
        correlation_id = generate_correlation_id()
        self._logger.info(f"Starting QC analysis", extra={"correlation_id": correlation_id})
        
        try:
            # Implementation here
            pass
        except Exception as e:
            self._logger.error(f"QC analysis failed", extra={
                "correlation_id": correlation_id,
                "error": str(e)
            })
            raise QualityAnalysisError(f"Analysis failed: {e}") from e
```

### Medical Device Specific Requirements
```python
# ALWAYS include audit trails for FDA compliance
@audit_trail(action="eeg_analysis")
async def analyze_eeg(self, study_id: str, user_id: str) -> AnalysisResult:
    """All actions affecting clinical decisions MUST be audited."""
    
    # Validate inputs rigorously
    if not study_id or not is_valid_uuid(study_id):
        raise ValidationError("Invalid study ID format")
    
    # Log PHI access (required for HIPAA)
    await self.audit_logger.log_phi_access(
        user_id=user_id,
        resource_id=study_id,
        action="read_eeg_data"
    )
    
    # Never log PHI in application logs
    self.logger.info(f"Processing study", extra={
        "study_id": study_id,  # OK - not PHI
        # "patient_name": name,  # NEVER do this
    })
```

### Error Handling Standards
```python
# Define specific exception types
class EEGProcessingError(Exception):
    """Base exception for EEG processing errors."""
    pass

class QualityAnalysisError(EEGProcessingError):
    """Raised when quality analysis fails."""
    pass

class ModelInferenceError(EEGProcessingError):
    """Raised when ML model inference fails."""
    pass

# Always handle errors gracefully
try:
    result = await self.model.predict(eeg_data)
except ModelInferenceError as e:
    # Log error with context
    self.logger.error("Model inference failed", extra={
        "error_type": type(e).__name__,
        "correlation_id": correlation_id
    })
    # Return safe default, never crash
    return AnalysisResult(
        status="failed",
        error="Analysis temporarily unavailable",
        fallback_action="manual_review_required"
    )
```

### Testing Requirements
```python
# Every module MUST have corresponding tests
# tests/test_module_name.py

import pytest
from hypothesis import given, strategies as st

class TestAutoRejectQualityAnalyzer:
    """Test coverage must be >80% for all modules."""
    
    @pytest.fixture
    def analyzer(self):
        """Use fixtures for dependency injection."""
        return AutorejectQualityAnalyzer(
            threshold_optimizer=MockOptimizer(),
            logger=MockLogger()
        )
    
    @given(
        n_channels=st.integers(min_value=1, max_value=64),
        bad_channel_ratio=st.floats(min_value=0, max_value=1)
    )
    async def test_analyze_with_variable_quality(
        self, analyzer, n_channels, bad_channel_ratio
    ):
        """Use property-based testing for robustness."""
        # Test implementation
        pass
    
    async def test_phi_logging_compliance(self, analyzer, audit_logger):
        """Verify HIPAA compliance in tests."""
        await analyzer.analyze_eeg("test_id", "user_id")
        
        # Verify audit trail created
        assert audit_logger.called_once()
        assert "patient_name" not in audit_logger.call_args
```

---

## Project Structure Rules

### File Organization
```
src/brain_go_brrr/
├── api/                    # FastAPI routes and HTTP handling
│   ├── v1/                # Version-specific endpoints
│   ├── dependencies.py    # Dependency injection setup
│   └── middleware.py      # Security, logging middleware
├── core/                  # Business logic (no frameworks)
│   ├── analyzers/         # Analysis implementations
│   ├── interfaces/        # Protocol definitions
│   └── models/           # Domain models
├── infrastructure/       # External service adapters
│   ├── database/         # Repository pattern only
│   ├── ml_models/        # Model loading and serving
│   └── storage/          # S3/filesystem adapters
└── shared/               # Cross-cutting concerns
    ├── logging.py        # Structured logging setup
    ├── security.py       # Auth/encryption utilities
    └── monitoring.py     # Metrics and tracing
```

### Import Rules
```python
# ALWAYS use absolute imports from src root
from brain_go_brrr.core.analyzers import AutorejectAnalyzer
from brain_go_brrr.shared.logging import get_logger

# NEVER use relative imports
# from ..analyzers import AutorejectAnalyzer  # Don't do this

# Group imports: standard lib, third-party, local
import asyncio
import logging
from pathlib import Path

import numpy as np
import torch
from mne import io

from brain_go_brrr.core.interfaces import IAnalyzer
```

---

## EEG-Specific Guidelines

### Data Handling
```python
# Always validate EEG data
def validate_eeg_data(raw: mne.io.Raw) -> None:
    """Validate EEG data meets processing requirements."""
    
    # Check sampling rate
    if raw.info['sfreq'] not in [128, 256, 512]:
        raise ValueError(f"Unsupported sampling rate: {raw.info['sfreq']}")
    
    # Check duration
    if raw.times[-1] < 60:  # Less than 1 minute
        raise ValueError("EEG recording too short for analysis")
    
    # Check channels
    required_channels = {'Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2'}
    available = set(raw.ch_names)
    if not required_channels.issubset(available):
        missing = required_channels - available
        raise ValueError(f"Missing required channels: {missing}")
```

### Model Integration
```python
# Load models safely with fallbacks
class EEGPTModelLoader:
    """Handle EEGPT model loading with safety checks."""
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._model = None
    
    async def load_model(self) -> None:
        """Load model with validation and fallback."""
        
        # Verify model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Check file integrity
        expected_hash = "..."  # From model card
        actual_hash = calculate_sha256(self.model_path)
        if actual_hash != expected_hash:
            raise ValueError("Model file corrupted")
        
        # Load with error handling
        try:
            self._model = torch.load(
                self.model_path,
                map_location=self.device
            )
            self._model.eval()  # Always set to eval mode
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to CPU if GPU fails
            if self.device == "cuda":
                self.device = "cpu"
                await self.load_model()  # Retry on CPU
            else:
                raise
```

---

## Security Guidelines

### Never Do This
```python
# NEVER hardcode credentials
# database_url = "postgresql://user:password@host/db"  # WRONG

# NEVER log sensitive data
# logger.info(f"Patient {patient_name} processed")  # WRONG

# NEVER store PHI in plain text
# cache[patient_id] = patient_data  # WRONG

# NEVER skip input validation
# study_id = request.query_params["id"]  # WRONG - validate first
```

### Always Do This
```python
# Use environment variables or secrets manager
database_url = os.getenv("DATABASE_URL")

# Log only non-sensitive identifiers
logger.info(f"Study {study_id} processed")

# Encrypt PHI at rest
encrypted_data = encrypt_pii(patient_data, key)
cache[patient_id] = encrypted_data

# Validate all inputs
study_id = validate_uuid(request.query_params.get("id"))
```

---

## Performance Guidelines

### Async Best Practices
```python
# Use async for I/O operations
async def process_batch(file_paths: List[Path]) -> List[Result]:
    """Process multiple files concurrently."""
    
    # Good: Concurrent processing
    tasks = [process_file(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle partial failures
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    if failed:
        logger.warning(f"Failed to process {len(failed)} files")
    
    return successful
```

### Memory Management
```python
# Handle large EEG files efficiently
def process_large_eeg(file_path: Path, chunk_size: int = 3600) -> None:
    """Process EEG in chunks to avoid memory issues."""
    
    with contextlib.closing(mne.io.read_raw_edf(file_path, preload=False)) as raw:
        n_samples = len(raw.times)
        
        for start in range(0, n_samples, chunk_size * int(raw.info['sfreq'])):
            end = min(start + chunk_size * int(raw.info['sfreq']), n_samples)
            
            # Load only current chunk
            chunk = raw.copy().crop(
                tmin=start/raw.info['sfreq'],
                tmax=end/raw.info['sfreq']
            ).load_data()
            
            # Process chunk
            process_chunk(chunk)
            
            # Explicitly free memory
            del chunk
```

---

## Documentation Standards

### API Documentation
```python
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_eeg(
    file: UploadFile = File(..., description="EEG file in EDF/BDF format"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> AnalysisResponse:
    """
    Analyze uploaded EEG file.
    
    This endpoint performs:
    1. File validation and virus scanning
    2. Quality control analysis
    3. Abnormality detection
    4. Event identification
    
    Returns:
        AnalysisResponse with job_id for tracking
        
    Raises:
        400: Invalid file format
        413: File too large
        500: Analysis service unavailable
    """
```

### Configuration Documentation
```yaml
# config/pipeline_config.yaml
# Pipeline configuration for EEGPT analysis

# Model configuration
models:
  eegpt:
    path: "/data/models/eegpt/pretrained/eegpt_large.pt"
    device: "cuda"  # Options: cuda, cpu
    batch_size: 32
    # Model expects 256Hz, 4-second windows
    expected_sfreq: 256
    window_duration: 4.0

# Quality control thresholds
quality_control:
  max_bad_channels_percent: 20  # Reject if >20% bad
  min_recording_duration: 60    # Minimum 60 seconds
  impedance_threshold: 50000    # 50kΩ max

# Clinical thresholds
clinical:
  abnormal_threshold: 0.7       # Flag as abnormal if >0.7
  urgent_threshold: 0.9         # Urgent review if >0.9
  min_confidence: 0.5           # Require manual review if <0.5
```

---

## Common Patterns

### Repository Pattern for Data Access
```python
class EEGStudyRepository:
    """Repository for EEG study data access."""
    
    def __init__(self, db: Database, storage: Storage):
        self._db = db
        self._storage = storage
    
    async def save_study(self, study: EEGStudy) -> str:
        """Save study metadata and file."""
        
        # Transaction pattern for consistency
        async with self._db.transaction() as tx:
            # Save metadata to database
            study_id = await tx.studies.insert(study.metadata)
            
            # Save file to storage
            file_path = await self._storage.save(
                study.file_data,
                f"studies/{study_id}/raw.edf"
            )
            
            # Update metadata with file path
            await tx.studies.update(study_id, {"file_path": file_path})
            
        return study_id
```

### Factory Pattern for Analyzers
```python
class AnalyzerFactory:
    """Create appropriate analyzer based on configuration."""
    
    @staticmethod
    def create_analyzer(analyzer_type: str, config: Config) -> IAnalyzer:
        """Factory method for analyzer creation."""
        
        analyzers = {
            "autoreject": AutorejectAnalyzer,
            "eegpt": EEGPTAnalyzer,
            "yasa": YASAAnalyzer
        }
        
        if analyzer_type not in analyzers:
            raise ValueError(f"Unknown analyzer: {analyzer_type}")
        
        analyzer_class = analyzers[analyzer_type]
        return analyzer_class(**config.get_analyzer_config(analyzer_type))
```

---

## Checklist for AI Agents

Before generating code, verify:
- [ ] Read relevant sections of literature-master-reference.md
- [ ] Understand PRD requirements being implemented
- [ ] Check TRD for technical constraints
- [ ] Review BDD scenarios for expected behavior
- [ ] Follow security guidelines (no PHI in logs)
- [ ] Include comprehensive docstrings
- [ ] Write corresponding tests
- [ ] Handle errors gracefully
- [ ] Use dependency injection
- [ ] Follow async patterns for I/O

---

## Getting Help

When in doubt:
1. Check `/docs/literature-master-reference.md` for technical details
2. Review existing code in similar modules
3. Ensure medical device compliance (when in doubt, be conservative)
4. Add comprehensive logging for debugging
5. Write tests to verify behavior

Remember: This is medical software. Patient safety is paramount. When uncertain, always choose the safer, more conservative approach.