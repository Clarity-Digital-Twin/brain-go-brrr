# CLAUDE.md - Brain-Go-Brrr Project (Enhanced Edition)

## 🧠 Critical Context

This is a medical-adjacent EEG analysis system using the EEGPT foundation model. While not FDA-approved, code quality matters - bugs could impact clinical decisions. Always prioritize safety and accuracy over speed.

## 📚 Project Overview

- **Purpose**: Production-ready Python wrapper around EEGPT for EEG analysis (QC, abnormality detection, sleep staging, event detection)
- **Model**: EEGPT 10M parameters at `/data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Architecture**: Service-oriented (not microservices yet), FastAPI + PyTorch + MNE
- **License**: Apache 2.0 (NOT MIT)
- **Python**: 3.11+ (current venv uses 3.13.2)

## 📖 Essential Reading Before Starting

```bash
# Technical specifications and requirements
/docs/literature-master-reference.md    # Complete technical reference
/docs/PRD-product-requirements.md      # What we're building
/docs/TRD-technical-requirements.md    # How we're building it
/docs/BDD-behavior-driven-development.md # Behavior scenarios
/docs/agentic-workflow.md              # AI-assisted development guide
/ROUGH_DRAFT.md                        # MVP implementation plan
```

## 🔧 Development Commands

```bash
# Environment Management
uv sync                    # Install/update dependencies
uv run python             # Run Python in project env
make dev-setup            # Full dev environment setup

# Quality Checks (ALWAYS run after changes)
make lint                 # Linting with ruff
make typecheck            # Type checking with mypy
make test                 # Run all tests
make format               # Auto-format code
make check-all            # Run all checks

# Development Workflow
make test-watch           # Watch mode for TDD
make coverage             # Test coverage report
make docs                 # Build documentation
```

## 🤖 GitHub Actions & Claude Bot

### Automated Development with Claude

This repository has Claude bot integration for autonomous development:

1. **Create issues with @claude tag** to trigger automatic implementation
2. **Comment @claude on PRs** for code reviews and improvements
3. **Claude creates PRs** with full implementations based on issue descriptions

### Usage Examples

```bash
# In a GitHub issue:
@claude implement this feature based on the issue description

# In a PR comment:
@claude review this code for security vulnerabilities

# For bug fixes:
@claude fix the TypeError in the payment processing module
```

### Requirements for Claude Bot

- Issues must have clear acceptance criteria
- Tag @claude in issue body or comments
- Claude follows all guidelines in this CLAUDE.md file
- PRs are created against the branch where issue was commented

## 🏗️ Architecture & Tech Stack

### Backend Stack

- **API**: FastAPI 0.100+ with Pydantic v2
- **ML**: PyTorch 2.0+, MNE-Python 1.6+, NumPy, SciPy
- **Models**: EEGPT, YASA (sleep), Autoreject (QC), tsfresh (features)
- **Queue**: Celery 5.3+ with Redis
- **DB**: PostgreSQL 15+ with TimescaleDB
- **Storage**: AWS S3 for EDF files and results

### Frontend Stack (Future)

- React 18+ / Next.js 14
- Material-UI / Ant Design
- Redux Toolkit for state
- Recharts for visualizations

## 📁 Project Structure

```
brain-go-brrr/
├── src/brain_go_brrr/      # Main package (underscore naming!)
│   ├── core/               # Configuration, logging, exceptions
│   ├── models/             # EEGPT & task-specific models
│   ├── data/               # Data loading & preprocessing
│   ├── training/           # Fine-tuning pipelines
│   ├── inference/          # Prediction & serving
│   └── cli.py              # Typer CLI interface
├── services/               # High-level processing services
│   ├── qc_flagger.py      # Autoreject + EEGPT quality control
│   ├── sleep_metrics.py   # YASA sleep staging & metrics
│   └── snippet_maker.py   # Snippet extraction + tsfresh
├── reference_repos/        # 9 critical EEG/ML libraries
│   ├── EEGPT/             # Original implementation
│   ├── mne-python/        # EEG processing foundation
│   ├── yasa/              # Sleep analysis
│   ├── autoreject/        # Artifact rejection
│   ├── braindecode/       # Deep learning for EEG
│   ├── pyEDFlib/          # Fast EDF/BDF reading
│   ├── mne-bids/          # BIDS format conversion
│   ├── tsfresh/           # Time-series features
│   └── tueg-tools/        # TUAB/TUEV dataset tools
├── data/                   # All data (gitignored)
│   ├── models/            # Model checkpoints
│   │   └── pretrained/    # EEGPT weights here
│   └── datasets/          # EEG datasets
│       └── external/
│           └── sleep-edf/ # 197 PSG recordings (✅ downloaded)
└── literature/            # Research papers & markdown
```

## 🧪 Core Features & Requirements

### 1. Quality Control Module

- Detect bad channels with >95% accuracy
- Calculate impedance metrics
- Identify artifacts (eye blinks, muscle, heartbeat)
- Generate reports in <30 seconds
- Implementation: `/services/qc_flagger.py`

### 2. Abnormality Detection ✅

- Binary classification (normal/abnormal)
- > 80% balanced accuracy target (AUROC ≥ 0.93)
- Confidence scoring (0-1)
- Triage flags: routine/expedite/urgent
- Reference: BioSerenity-E1 (94.63% accuracy)
- **IMPLEMENTED**: Linear probe training in `experiments/eegpt_linear_probe/`
- **FIXED**: Channel mapping (T3→T7, T4→T8, T5→P7, T6→P8)
- **STATUS**: Training with 20 channels, 8s windows @ 256Hz

### 3. Event Detection

- Epileptiform discharge identification
- GPED/PLED pattern detection
- Time-stamped events with confidence
- Implementation pending

### 4. Sleep Analysis

- 5-stage classification (W, N1, N2, N3, REM)
- Hypnogram visualization
- Sleep metrics: efficiency, REM%, N3%, WASO
- Implementation: `/services/sleep_metrics.py`
- Reference: YASA (87.46% accuracy)

## 🎯 Performance Targets

- Process 20-minute EEG in <2 minutes
- Support 50 concurrent analyses
- API response time <100ms
- 99.5% uptime SLA
- Handle files up to 2GB

## 🔬 EEG Processing Standards

### EEGPT Specifications

- **Sampling**: 256 Hz (resample if needed)
- **Windows**: 8 seconds (2048 samples) for TUAB linear probe
- **Channels**: 20 standard channels (modern naming)
- **Patch size**: 64 samples (250ms)
- **Architecture**: Vision Transformer with masked autoencoding

### ⚠️ CRITICAL: Channel Mapping

TUAB uses OLD naming → Convert to MODERN naming:
- T3 → T7
- T4 → T8
- T5 → P7
- T6 → P8

**Current standard channels (20)**: FP1, FP2, F7, F3, FZ, F4, F8, T7, C3, CZ, C4, T8, P7, P3, PZ, P4, P8, O1, O2, OZ

### Processing Pipeline

```python
# Standard pipeline order:
Raw EEG → Autoreject (QC) → EEGPT (Features) → Task Head (Prediction)

# Filtering standards:
- Bandpass: 0.5-50 Hz typical
- Notch: 50/60 Hz based on region
- Z-score normalization per channel
```

### Channel Standards

- Use 10-20 system naming (Fp1, Fp2, C3, C4, etc.)
- Handle missing channels gracefully
- Minimum channels for analysis: 19
- Reference: average or linked mastoids

## 💻 Code Style Rules

```python
# ALWAYS use type hints
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy.typing as npt

def process_eeg(
    file_path: Path,
    sampling_rate: int = 256,
    window_size: float = 4.0
) -> Dict[str, npt.NDArray[np.float32]]:
    """Process EEG data with EEGPT.

    Args:
        file_path: Path to EDF file
        sampling_rate: Target sampling rate in Hz
        window_size: Window size in seconds

    Returns:
        Dictionary with predictions and confidence scores
    """
    # Implementation here
```

### Key Principles

- Use `pathlib.Path`, NEVER string paths
- Async/await for ALL I/O operations
- Dependency injection (SOLID principles)
- Never hardcode paths - use config
- Comprehensive error handling
- Log errors, but NEVER log PHI/patient data

## 🧪 Testing Philosophy

### Test-Driven Development (TDD)

```bash
# 1. Write failing test first
uv run pytest tests/test_new_feature.py::test_specific -xvs

# 2. Implement minimal code to pass
# 3. Refactor for clarity
# 4. Ensure all tests still pass
make test

# Watch mode for rapid iteration
make test-watch
```

### Test Structure

```python
# tests/test_sleep_analysis.py
import pytest
from pathlib import Path
import numpy as np

@pytest.fixture
def mock_eeg_data():
    """Create realistic mock EEG data."""
    return np.random.randn(19, 256 * 300)  # 19 channels, 5 minutes

def test_sleep_staging_accuracy(mock_eeg_data):
    """Test sleep staging meets accuracy requirements."""
    # Given: Clean EEG data
    # When: Running sleep analysis
    # Then: Accuracy > 80%
```

## 🚀 API Development

### Endpoint Structure

```python
# FastAPI with full typing
from fastapi import APIRouter, UploadFile, BackgroundTasks
from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    file_id: str = Field(..., description="S3 file ID")
    analysis_type: Literal["qc", "sleep", "abnormal", "events"]
    options: Dict[str, Any] = Field(default_factory=dict)

@router.post("/api/v1/eeg/analyze")
async def analyze_eeg(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """Queue EEG analysis job."""
    # Implementation
```

### Security Requirements

- OAuth2 with JWT (RS256)
- HIPAA compliant data handling
- Encryption: AES-256 (rest), TLS 1.3 (transit)
- Audit logging for all operations
- Data retention: 7 years

## 📋 First Vertical Slice (MVP)

Based on ROUGH_DRAFT.md, implement in this order:

### 1. Basic EEG Pipeline

```bash
# Test basic model loading and inference
uv run python scripts/test_sleep_analysis.py

# Create core pipeline test
think hard about implementing the basic EEG processing pipeline
write tests for loading EDF, running EEGPT, getting features
```

### 2. Sleep Analysis Service

- Load Sleep-EDF data ✅ (already downloaded)
- Run YASA sleep staging
- Compare with EEGPT features
- Generate hypnogram
- Calculate sleep metrics

### 3. Quality Control Service

- Implement Autoreject integration
- Add EEGPT-based QC
- Generate QC reports
- Test on noisy data

### 4. Simple API

- FastAPI endpoint for file upload
- Async processing with Celery
- Status checking endpoint
- Result retrieval

## 🛠️ Common Workflows

### Adding a New Feature

```bash
# 1. Think and plan
think hard about implementing [FEATURE]

# 2. Create comprehensive tests
"Write tests for [FEATURE] including:
- Normal operation
- Edge cases (short recordings, missing channels)
- Error handling
- Performance benchmarks"

# 3. Implement following patterns
"Implement [FEATURE] using service pattern.
Reference /services/qc_flagger.py for structure"

# 4. Verify and document
make test
make docs
```

### Processing Sleep-EDF Data

```python
from pathlib import Path
import mne
from services.sleep_metrics import SleepAnalyzer

# Load data
edf_path = Path("data/datasets/external/sleep-edf/sleep-cassette/SC4001E0-PSG.edf")
raw = mne.io.read_raw_edf(edf_path, preload=True)

# Run analysis
analyzer = SleepAnalyzer()
results = analyzer.run_full_sleep_analysis(raw)

# Results include:
# - hypnogram (30s epochs)
# - sleep_efficiency (%)
# - sleep_stages (percentages)
# - total_sleep_time (minutes)
```

## 🚨 Safety & Compliance

### Critical Safety Rules

1. **Input Validation**: Always validate EEG data format and ranges
2. **Confidence Scores**: Never present results without confidence
3. **Error Handling**: Fail safely with informative messages
4. **Audit Trail**: Log all predictions with timestamps
5. **Human Review**: Flag low-confidence results

### HIPAA Compliance

- No PHI in logs or error messages
- Secure all data at rest and in transit
- Implement access controls
- Maintain audit logs
- Follow 21 CFR Part 11 for electronic records

## 🐛 Debugging Tools

```bash
# Check GPU availability
uv run python -c "import torch; print(torch.cuda.is_available())"

# Test EEGPT model loading
uv run python -c "
from pathlib import Path
import torch
model_path = Path('data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt')
print(f'Model exists: {model_path.exists()}')
print(f'Model size: {model_path.stat().st_size / 1e6:.1f} MB')
"

# Profile memory usage
uv run python -m memory_profiler scripts/test_sleep_analysis.py

# Check Sleep-EDF data
find data/datasets/external/sleep-edf -name "*.edf" | wc -l
# Should show 397 files
```

## 📊 Performance Benchmarks

Target metrics from literature:

- **EEGPT**: 65-87.5% accuracy across tasks
- **Sleep Staging**: 87.46% (YASA), 85% (EEGPT)
- **Abnormal Detection**: 94.63% (BioSerenity-E1)
- **Artifact Rejection**: 87.5% expert agreement

## 🎯 When to Use Reference Repos

- **EEGPT**: Model architecture and training code
- **MNE-Python**: EEG data loading and preprocessing
- **YASA**: Sleep staging algorithms
- **Autoreject**: Artifact detection methods
- **Braindecode**: Deep learning utilities
- **tsfresh**: Time-series feature extraction
- **pyEDFlib**: Fast EDF file reading
- **mne-bids**: BIDS format conversion

## 📝 Git Workflow

```bash
# Feature branches
git checkout -b feature/add-event-detection

# Commit format
git commit -m "feat(events): add epileptiform discharge detection

- Implement sliding window approach
- Add confidence thresholding
- Include temporal clustering
"

# Before pushing
make check-all  # Run all quality checks
```

## 🤔 Thinking Triggers

- `think` - Standard analysis
- `think hard` - Complex architectural decisions
- `think harder` - Multi-component integration
- `ultrathink` - System-wide changes

## ❌ Do NOT

- Log patient identifiers or PHI
- Hardcode file paths
- Skip input validation
- Ignore error handling
- Make synchronous I/O calls
- Modify data in-place
- Create files without user request
- Use print() for debugging (use logging)
- Assume GPU is available
- Trust external data without validation

## 🎯 Next Steps for First Vertical Slice

1. **Test Model Loading** ✅ (script exists)
2. **Create Basic Pipeline Test** (TDD approach)
3. **Implement Core Processing** (EDF → EEGPT → Features)
4. **Add Sleep Analysis** (using YASA service)
5. **Create Simple API** (FastAPI with one endpoint)
6. **Add Integration Test** (end-to-end flow)

Remember: This handles brain data. Accuracy matters. Test everything.

## 🔗 Quick Links

- [Literature Master Reference](docs/literature-master-reference.md)
- [Product Requirements](docs/PRD-product-requirements.md)
- [Technical Requirements](docs/TRD-technical-requirements.md)
- [Agentic Workflow Guide](docs/agentic-workflow.md)
- [Project Status](PROJECT_STATUS.md)
