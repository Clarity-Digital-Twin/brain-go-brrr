# Development Standards & Best Practices
## EEGPT Clinical Decision Support System

### Document Control
- **Version**: 1.0.0
- **Status**: Active
- **Last Updated**: 2025-01-17
- **Enforcement**: Required for all contributors

---

## Overview

This document establishes coding standards, development practices, and quality requirements for the EEGPT project. All code must comply with these standards to ensure safety, reliability, and maintainability of our medical device software.

### Guiding Principles
1. **Patient Safety First** - When in doubt, choose the safer approach
2. **Regulatory Compliance** - Meet FDA, HIPAA, and international standards
3. **Code as Documentation** - Self-documenting, clear code
4. **Test Everything** - Comprehensive testing at all levels
5. **Fail Gracefully** - Never crash, always provide safe defaults

---

## Code Organization

### Project Structure
```
brain-go-brrr/
├── .claude/                # AI agent guidelines
│   └── guidelines.md      # AI-specific coding standards
├── .github/               # GitHub configurations
│   ├── workflows/         # CI/CD pipelines
│   └── CODEOWNERS        # Code ownership
├── docs/                  # Project documentation
│   ├── PRD-*.md          # Product requirements
│   ├── TRD-*.md          # Technical requirements
│   ├── BDD-*.md          # Behavior specifications
│   └── *.md              # Other documentation
├── src/                   # Source code
│   └── brain_go_brrr/    # Main package
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── scripts/              # Utility scripts
├── config/               # Configuration files
└── data/                 # Data directories
    ├── models/           # Model weights
    └── datasets/         # Test datasets
```

### Module Organization (SOLID Principles)
```python
# Single Responsibility Principle
# Each module has ONE reason to change

# core/analyzers/qc_analyzer.py
class QualityAnalyzer:
    """Responsible ONLY for quality analysis logic."""
    
# infrastructure/storage/s3_adapter.py
class S3StorageAdapter:
    """Responsible ONLY for S3 operations."""

# api/routers/analysis.py
class AnalysisRouter:
    """Responsible ONLY for HTTP request handling."""
```

---

## Coding Standards

### Python Style Guide

#### General Rules
```python
# File header template (REQUIRED for all Python files)
"""
Module: descriptive_module_name.py
Purpose: One-line description of module purpose
Created: YYYY-MM-DD
Modified: YYYY-MM-DD

Requirements:
    - PRD: FR1.1, FR2.3 (link to specific requirements)
    - TRD: Section 3.2.1

Usage:
    from brain_go_brrr.module import MainClass
    
    instance = MainClass(config)
    result = await instance.process(data)
"""

# Import order (enforced by isort)
# 1. Standard library
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

# 2. Third-party packages
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

# 3. Local application
from brain_go_brrr.core.interfaces import IAnalyzer
from brain_go_brrr.shared.logging import get_logger
```

#### Naming Conventions
```python
# Classes: PascalCase
class EEGAnalyzer:
    pass

# Functions/Methods: snake_case
def analyze_eeg_quality(raw_data: np.ndarray) -> float:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_CHANNELS = 64
DEFAULT_SAMPLING_RATE = 256

# Private: Leading underscore
class Service:
    def __init__(self):
        self._internal_state = {}
    
    def _private_method(self):
        pass

# Interfaces: Prefix with 'I'
class IQualityAnalyzer(Protocol):
    pass
```

#### Type Annotations (Required)
```python
# Always use type hints
def process_eeg(
    file_path: Path,
    sampling_rate: int = 256,
    channels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process EEG file with specified parameters."""
    pass

# Use TypedDict for complex structures
from typing import TypedDict

class AnalysisResult(TypedDict):
    job_id: str
    status: str
    confidence: float
    events: List[Dict[str, Any]]
```

### JavaScript/TypeScript Standards

#### React Component Structure
```typescript
// components/EEGViewer/EEGViewer.tsx

import React, { useState, useEffect, useCallback } from 'react';
import { Box, Typography } from '@mui/material';
import { useAppSelector, useAppDispatch } from '@/hooks/redux';
import type { EEGData, AnalysisResult } from '@/types';

interface EEGViewerProps {
  data: EEGData;
  onAnalysisComplete?: (result: AnalysisResult) => void;
}

/**
 * EEG waveform viewer component
 * Displays multi-channel EEG data with zoom/pan capabilities
 */
export const EEGViewer: React.FC<EEGViewerProps> = ({
  data,
  onAnalysisComplete,
}) => {
  const [selectedChannel, setSelectedChannel] = useState<number>(0);
  const dispatch = useAppDispatch();
  
  // Component implementation
  
  return (
    <Box data-testid="eeg-viewer">
      {/* Component JSX */}
    </Box>
  );
};
```

---

## Testing Standards

### Test Coverage Requirements
- **Unit Tests**: Minimum 80% coverage
- **Integration Tests**: All API endpoints
- **E2E Tests**: Critical user journeys
- **Performance Tests**: All data processing pipelines

### Test Structure
```python
# tests/unit/test_qc_analyzer.py

import pytest
from hypothesis import given, strategies as st
from brain_go_brrr.core.analyzers import QualityAnalyzer

class TestQualityAnalyzer:
    """Test suite for QualityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return QualityAnalyzer(config=test_config)
    
    @pytest.mark.parametrize("n_channels,expected", [
        (19, True),   # Standard 10-20 system
        (32, True),   # Extended montage
        (5, False),   # Too few channels
    ])
    def test_channel_validation(self, analyzer, n_channels, expected):
        """Test channel count validation."""
        result = analyzer.validate_channels(n_channels)
        assert result == expected
    
    @given(
        bad_ratio=st.floats(min_value=0, max_value=1),
        threshold=st.floats(min_value=0, max_value=1)
    )
    def test_quality_threshold(self, analyzer, bad_ratio, threshold):
        """Property-based test for quality thresholds."""
        result = analyzer.assess_quality(bad_ratio, threshold)
        
        # Properties that must always hold
        assert 0 <= result.score <= 1
        assert result.passed == (bad_ratio <= threshold)
```

### BDD Test Implementation
```python
# tests/bdd/test_file_upload.py

from pytest_bdd import scenarios, given, when, then
import pytest

# Load all scenarios from feature file
scenarios('../features/file_upload.feature')

@given('I am authenticated as an EEG technologist')
def authenticated_user(client, auth_token):
    client.headers['Authorization'] = f'Bearer {auth_token}'

@when('I upload the file through the web interface')
def upload_file(client, test_file):
    response = client.post(
        '/api/v1/eeg/upload',
        files={'file': test_file}
    )
    return response

@then('I should receive a job ID')
def check_job_id(upload_response):
    assert upload_response.status_code == 202
    assert 'job_id' in upload_response.json()
```

---

## Security Standards

### Authentication & Authorization
```python
# Use dependency injection for auth
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Validate JWT token and return current user."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = await get_user(username=payload.get("sub"))
        if user is None:
            raise HTTPException(status_code=401)
        return user
    except JWTError:
        raise HTTPException(status_code=401)

# Apply to routes
@router.get("/protected")
async def protected_route(user: User = Depends(get_current_user)):
    return {"user": user.username}
```

### Data Protection
```python
# NEVER log PHI
logger.info(f"Processing study", extra={
    "study_id": study_id,      # OK - anonymized ID
    # "patient_name": name,    # NEVER do this
    # "mrn": medical_record,   # NEVER do this
})

# Encrypt sensitive data at rest
from cryptography.fernet import Fernet

def encrypt_phi(data: str, key: bytes) -> bytes:
    """Encrypt PHI before storage."""
    f = Fernet(key)
    return f.encrypt(data.encode())

# Sanitize all inputs
from brain_go_brrr.shared.validators import sanitize_input

@router.post("/analyze")
async def analyze(
    study_id: str = Query(..., regex="^[a-f0-9-]{36}$")  # UUID only
):
    clean_id = sanitize_input(study_id)
    # Process with sanitized input
```

---

## Performance Standards

### Async/Await Best Practices
```python
# Good: Concurrent I/O operations
async def process_batch(file_ids: List[str]) -> List[Result]:
    """Process multiple files concurrently."""
    tasks = [process_file(fid) for fid in file_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle partial failures
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [(fid, r) for fid, r in zip(file_ids, results) 
              if isinstance(r, Exception)]
    
    if failed:
        logger.warning(f"Failed to process {len(failed)} files")
    
    return successful

# Bad: Sequential processing
async def process_batch_bad(file_ids: List[str]) -> List[Result]:
    results = []
    for fid in file_ids:  # Don't do this
        result = await process_file(fid)
        results.append(result)
    return results
```

### Memory Management
```python
# Stream large files instead of loading entirely
async def process_large_eeg(file_path: Path) -> None:
    """Process large EEG files in chunks."""
    
    CHUNK_DURATION = 60  # seconds
    
    with mne.io.read_raw_edf(file_path, preload=False) as raw:
        sfreq = raw.info['sfreq']
        n_samples = len(raw.times)
        chunk_samples = int(CHUNK_DURATION * sfreq)
        
        for start_idx in range(0, n_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, n_samples)
            
            # Load only current chunk
            chunk = raw[:, start_idx:end_idx][0]
            
            # Process chunk
            await process_chunk(chunk)
            
            # Explicit cleanup
            del chunk
```

---

## Documentation Standards

### API Documentation
```python
from fastapi import APIRouter, File, UploadFile
from typing import List

router = APIRouter()

@router.post(
    "/analyze",
    summary="Analyze EEG Recording",
    description="""
    Performs comprehensive analysis of uploaded EEG file.
    
    The analysis includes:
    - Quality control assessment
    - Abnormality detection
    - Event identification
    - Sleep staging (if applicable)
    
    Processing is asynchronous - returns job_id for status polling.
    """,
    response_model=AnalysisResponse,
    responses={
        202: {"description": "Analysis job accepted"},
        400: {"description": "Invalid file format"},
        413: {"description": "File too large"},
        500: {"description": "Internal server error"}
    },
    tags=["analysis"]
)
async def analyze_eeg(
    file: UploadFile = File(
        ...,
        description="EEG file in EDF or BDF format"
    ),
    priority: str = Query(
        "normal",
        regex="^(normal|high|urgent)$",
        description="Processing priority"
    )
) -> AnalysisResponse:
    """
    Analyze uploaded EEG file.
    
    Args:
        file: EEG recording file (EDF/BDF format)
        priority: Processing priority level
        
    Returns:
        AnalysisResponse with job_id and initial status
        
    Raises:
        HTTPException: If file validation fails
    """
    # Implementation
```

### Code Comments
```python
# Good: Explain WHY, not WHAT
# Use 75% confidence threshold per clinical validation study (REF-2024-001)
ABNORMAL_THRESHOLD = 0.75

# Bad: Redundant comment
# Set threshold to 0.75
ABNORMAL_THRESHOLD = 0.75

# Good: Complex algorithm explanation
def optimize_threshold(data: np.ndarray) -> float:
    """
    Optimize detection threshold using Youden's J statistic.
    
    This maximizes sensitivity + specificity - 1, providing
    optimal balance for clinical screening. See Youden 1950.
    
    Clinical validation showed J=0.82 at threshold=0.75.
    """
    # Implementation
```

---

## Version Control Standards

### Branch Naming
```bash
# Feature branches
feature/add-sleep-staging
feature/JIRA-123-event-detection

# Bug fixes
bugfix/fix-memory-leak
bugfix/JIRA-456-validation-error

# Releases
release/v1.2.0
hotfix/v1.2.1
```

### Commit Messages
```bash
# Format: <type>(<scope>): <subject>
#
# Types: feat, fix, docs, style, refactor, test, chore
# Scope: api, core, ml, frontend, etc.

# Good examples
feat(api): add sleep staging endpoint
fix(ml): resolve memory leak in batch processing
docs(readme): update installation instructions
test(qc): add edge cases for channel validation

# Bad examples
"fix bug"              # Too vague
"UPDATES"             # Not descriptive
"wip"                 # Don't commit work in progress
```

### Pull Request Standards
```markdown
## Description
Brief description of changes and why they're needed.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that breaks existing functionality)

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No sensitive data in logs
- [ ] Security review if auth/crypto changes

## Related Issues
Closes #123
Relates to #456
```

---

## Continuous Integration

### Required Checks
```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run linters
        run: |
          uv run ruff check .
          uv run mypy src/
          uv run black --check .

  test:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests
        run: |
          uv run pytest tests/ \
            --cov=brain_go_brrr \
            --cov-report=xml \
            --cov-fail-under=80

  security:
    runs-on: ubuntu-latest
    steps:
      - name: Security scan
        run: |
          uv run bandit -r src/
          uv run safety check
          trivy fs --security-checks vuln .
```

### Definition of Done
- [ ] Code complete and follows standards
- [ ] Unit tests written and passing (>80% coverage)
- [ ] Integration tests updated if needed
- [ ] Documentation updated
- [ ] Code reviewed by 2+ team members
- [ ] Security scan passing
- [ ] Performance benchmarks met
- [ ] Deployed to staging environment
- [ ] Acceptance criteria verified

---

## Monitoring & Logging

### Structured Logging
```python
import structlog

logger = structlog.get_logger()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
)

# Use structured logging
logger.info(
    "analysis_complete",
    study_id=study_id,
    duration=processing_time,
    model_version="1.2.0",
    abnormal_probability=0.82,
    correlation_id=correlation_id
)
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
analysis_counter = Counter(
    'eeg_analyses_total',
    'Total number of EEG analyses',
    ['status', 'model_version']
)

processing_time = Histogram(
    'eeg_processing_duration_seconds',
    'Time spent processing EEG files',
    ['file_size_category']
)

active_jobs = Gauge(
    'eeg_active_jobs',
    'Number of currently processing jobs'
)

# Use metrics
@processing_time.time()
async def analyze():
    active_jobs.inc()
    try:
        # Process
        analysis_counter.labels(status='success', model_version='1.2').inc()
    finally:
        active_jobs.dec()
```

---

## Compliance Checklist

### Every Code Change Must:
- [ ] Follow coding standards
- [ ] Include appropriate tests
- [ ] Update documentation
- [ ] Pass security scan
- [ ] Consider HIPAA compliance
- [ ] Log appropriately (no PHI)
- [ ] Handle errors gracefully
- [ ] Maintain audit trail
- [ ] Use dependency injection
- [ ] Follow SOLID principles

### Before Release:
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Security review done
- [ ] Performance validated
- [ ] Compliance verified
- [ ] Rollback plan ready
- [ ] Monitoring configured
- [ ] Alerts configured

This document is enforced through automated tooling and code review. Non-compliant code will be rejected by CI/CD pipeline.