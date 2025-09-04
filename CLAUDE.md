# CLAUDE.md - Brain-Go-Brrr Project (FIXED ARCHITECTURE)

## 🔥🔥🔥 CRITICAL: ARCHITECTURE RULES TO PREVENT DISASTERS 🔥🔥🔥

### RULE #1: NO PARALLEL IMPLEMENTATIONS EVER
**NEVER CREATE DUPLICATE CODE IN experiments/ AND src/**

#### What Went Wrong (The Disaster):
- Created TWO PARALLEL UNIVERSES that don't communicate
- experiments/ reimplemented everything from scratch
- src/ had working components that were ignored
- Result: AUROC=0.50 (complete training failure), wasted compute, confusion

#### The ONLY Correct Architecture:
```python
# experiments/train_anything.py - MUST BE THIN
from brain_go_brrr.infra.data import Dataset  # ALWAYS USE SRC
from brain_go_brrr.infra.ml_models import Model  # NEVER REIMPLEMENT
from brain_go_brrr.domain.preprocessing import preprocess  # REUSE!

# FORBIDDEN: Creating new datasets, models, preprocessing in experiments/
```

### RULE #2: CHECK BEFORE BUILDING
1. **ALWAYS** search src/ for existing implementations
2. **NEVER** build without checking what exists
3. **ALWAYS** reuse components from src/
4. **NEVER** create "isolated" implementations

### RULE #3: NORMALIZATION IS CRITICAL
- MNE outputs: 1e-5 scale (Volts)
- EEGPT expects: N(0,1) normalized
- ALWAYS normalize before model input
- NEVER trust raw sensor data

## 🚨 Current Architecture Status (Aug 28, 2025)

**PROBLEM DISCOVERED**: Parallel implementations in experiments/ and src/
- Status: BOTH FIXED with normalization
- TODO: Migrate experiments/ to use src/ components

## 🚨 CRITICAL WARNING: PyTorch Lightning 2.5.2 Bug

**DO NOT USE PYTORCH LIGHTNING FOR TRAINING!** Lightning 2.5.2 has a critical bug that causes training to hang indefinitely at:
```
Loading `train_dataloader` to estimate number of stepping batches
```

This occurs with large cached datasets (>100k samples) and CANNOT be fixed with any settings. We tried:
- `deterministic=False` ❌
- `limit_train_batches` as integer ❌
- `max_steps=10000` ❌
- `num_sanity_val_steps=0` ❌
- `fast_dev_run=True` ❌

**SOLUTION**: Use `experiments/eegpt_linear_probe/train_tuab.py` (pure PyTorch)

## 🧠 Critical Context

This is a medical-adjacent EEG analysis system using the EEGPT foundation model. While not FDA-approved, code quality matters - bugs could impact clinical decisions. Always prioritize safety and accuracy over speed.

## ✅ What's Currently Working
- **YASA Sleep Analysis**: 100% functional, 87% accuracy
- **Autoreject QC**: Bad channel detection, artifact rejection
- **EEGPT Features**: 2,048-dim features (4×512 summary tokens, flattened)
- **FastAPI Server**: REST API with Redis caching
- **CI/CD Pipeline**: All branches green, pre-commit hooks fixed
- **Unit Tests**: All tests pass locally (run `make test`)
- **Architecture**: Unified - experiments/ uses src/ components
- **Normalization**: SSOT in wrapper, datasets emit Volts (SI units)
- **Channel Validation**: Enforces correct order per dataset

## 🟡 In Progress
- **TUAB Abnormality Detection**: Training linear probe
- **experiments/ cleanup**: Removing remaining sys.path.insert hacks

## ❌ Not Implemented
- **Event Detection**: Architecture docs only, no code
- **Authentication**: No OAuth2/JWT
- **Production Deployment**: No Kubernetes/PostgreSQL
- **Message Queue**: No Celery implementation

## 📚 Project Overview

- **Purpose**: Production-ready Python wrapper around EEGPT for EEG analysis (QC, abnormality detection, sleep staging, event detection)
- **Model**: EEGPT 10M parameters at `/data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`
- **Architecture**: Service-oriented (not microservices yet), FastAPI + PyTorch + MNE
- **License**: Apache 2.0 (NOT MIT)
- **Python**: 3.11-3.12 (3.13 not supported due to scipy/OpenBLAS issues)

## 📖 Essential Documentation

```bash
# Consolidated, accurate documentation (6 files total)
/docs/README.md           # Documentation hub - START HERE
/docs/QUICK_START.md     # Get running in 5 minutes
/docs/ARCHITECTURE.md    # System design (actual implementation)
/docs/API.md            # REST endpoints (working only)
/docs/TRAINING.md       # Model training guide
/docs/TESTING.md        # Testing guidelines

# Root documentation
/README.md              # Project overview
/CLAUDE.md             # This file - AI assistant context
```

## 🔧 Development Commands

### 🚨 CRITICAL: Pre-Push Validation (MUST PASS OR CI WILL FAIL)

```bash
# ALWAYS RUN THESE BEFORE PUSHING (uses project Makefile targets)
make format      # Auto-format code
make lint        # Check for lint errors
make typecheck   # Run type checking
make validate    # Full pre-push validation suite

# Or run all checks at once:
make check-all   # Runs all CI checks locally
```

### 🔒 SECURITY: torch.load Requirements

**CI/CD WILL FAIL if torch.load is used unsafely!** All `torch.load` calls MUST either:
1. Use `weights_only=True` for pure tensor data
2. Use `weights_only=False` WITH `# nosec:weights_only` comment explaining why unsafe loading is needed
3. Use the `safe_load` wrapper from `brain_go_brrr.infra.safe_load`

```python
# ✅ GOOD - Safe loading for tensors
checkpoint = torch.load(path, map_location='cpu', weights_only=True)

# ✅ GOOD - Unsafe with explicit justification
data = torch.load(cache_file, weights_only=False)  # nosec:weights_only - cache contains numpy arrays

# ❌ BAD - Will fail CI/CD
data = torch.load(path)  # Missing weights_only parameter
```

### Standard Development Commands

```bash
# Environment Management
uv sync                    # Install/update dependencies
uv run python             # Run Python in project env
make dev-setup            # Full dev environment setup

# Quality Checks (safe wrappers for CI commands)
make format               # Auto-format with CI's ruff
make lint                 # Linting with CI's ruff
make typecheck            # Type checking with CI's mypy
make test                 # Run all tests
make check-all            # Run ALL CI checks locally

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

## 🛑 ARCHITECTURE COMMANDMENTS (NEVER VIOLATE)

### COMMANDMENT 1: src/ is the SOURCE OF TRUTH
- ALL reusable components go in src/
- experiments/ MUST import from src/
- NO reimplementing what exists in src/

### COMMANDMENT 2: experiments/ is THIN
- ONLY training loops and configs
- IMPORTS everything else from src/
- If you're writing >100 lines, you're doing it wrong

### COMMANDMENT 3: Check Before Creating
```bash
# BEFORE creating ANY new file:
grep -r "class.*Dataset" src/  # Check for existing datasets
grep -r "def.*preprocess" src/  # Check for existing preprocessing
grep -r "class.*Model" src/  # Check for existing models
```

### COMMANDMENT 4: One Implementation Per Function
- ONE dataset class per dataset type
- ONE preprocessing pipeline per data type
- ONE model wrapper per model
- DELETE duplicates immediately

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

## 📁 Project Structure (CORRECT USAGE)

### ⚠️ CRITICAL: How Each Directory MUST Be Used

**src/brain_go_brrr/** - ALL REUSABLE CODE
- ✅ Datasets, models, preprocessing, utils
- ✅ Anything used by multiple scripts
- ❌ NEVER duplicate what's here

**experiments/** - TRAINING SCRIPTS ONLY
- ✅ Training loops, experiment configs
- ✅ Paper reproduction scripts
- ❌ NEVER reimplement datasets/models/preprocessing
- ❌ MUST import from src/

## 📁 Project Structure

```
brain-go-brrr/
├── src/brain_go_brrr/      # Main package (underscore naming!)
│   ├── api/                # FastAPI routes and endpoints
│   ├── application/        # Application layer (use cases, services)
│   ├── domain/             # Domain layer (entities, business logic)
│   ├── infra/              # Infrastructure layer (external services, ML models)
│   │   └── ml_models/      # EEGPT models and wrappers
│   ├── services/           # Service layer (orchestration)
│   ├── models/             # Legacy models (being migrated)
│   ├── utils/              # Utilities and helpers
│   └── cli.py              # Typer CLI interface
├── experiments/            # Training scripts and experiments
│   └── eegpt_linear_probe/ # EEGPT fine-tuning (USE THIS FOR TRAINING)
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
│       ├── tuab/          # TUH Abnormal dataset v3.0.1
│       ├── tuev/          # TUH Events dataset v2.0.1
│       └── sleep-edf/     # Sleep-EDF dataset (197 recordings)
└── literature/            # Research papers & markdown
```

## 🧪 Core Features & Requirements

### 1. Quality Control Module

- Detect bad channels with >95% accuracy
- Calculate impedance metrics
- Identify artifacts (eye blinks, muscle, heartbeat)
- Generate reports in <30 seconds
- Domain: `src/brain_go_brrr/domain/quality/controller.py`
- Infrastructure: `src/brain_go_brrr/infra/preprocessing/autoreject_adapter.py`

### 2. Abnormality Detection 🟢 TRAINING

- Binary classification (normal/abnormal)
- **Target AUROC**: ≥ 0.869 (paper performance)
- Confidence scoring (0-1)
- Triage flags: routine/expedite/urgent
- **CRITICAL**: Must use 4-second windows (EEGPT pretrained on 4s)
- **FEATURES**: 2,048 dimensions (4 summary tokens × 512, flattened) - NOT 512, NOT 32,768
- **IMPLEMENTED**: Linear probe training in `experiments/eegpt_linear_probe/`
- **FIXED**: Channel mapping (T3→T7, T4→T8, T5→P7, T6→P8)
- **STATUS**: Training with 4s windows @ 256Hz - `tmux attach -t tuab_training`

### 3. Event Detection

- Epileptiform discharge identification
- GPED/PLED pattern detection
- Time-stamped events with confidence
- Implementation pending

### 4. Sleep Analysis ✅ WORKING

- 5-stage classification (W, N1, N2, N3, REM)
- Hypnogram visualization
- Sleep metrics: efficiency, REM%, N3%, WASO
- Implementation: `/src/brain_go_brrr/infra/external/yasa_adapter.py`
- Reference: YASA (87.46% accuracy)

#### ⚠️ CRITICAL: Channel Aliasing for Non-Standard Montages
YASA works with ANY channel count but prefers central channels (C3/C4). For datasets like Sleep-EDF that use non-standard montages, our adapter includes automatic channel aliasing:

```python
# Automatic aliasing for non-standard montages (e.g., Sleep-EDF)
DEFAULT_ALIASES = {
    "EEG Fpz-Cz": "C4",  # Frontal→Central (Sleep-EDF specific)
    "EEG Pz-Oz": "O2",   # Parietal→Occipital (Sleep-EDF specific)
    "Fpz": "C3",         # Single electrode mapping
}

# Usage - works with ANY channel count
stager = YASASleepStager()
results = stager.stage_sleep(raw)  # Auto-selects best channel
```

**Key Facts**:
- YASA achieves 85%+ accuracy with just 1 central EEG channel
- With aliasing, accuracy improves to ~87% for non-standard montages
- NOT limited to 2 channels - that's just Sleep-EDF's configuration

## 🎯 Performance Targets

- Process 20-minute EEG in <2 minutes
- Support 50 concurrent analyses
- API response time <100ms
- 99.5% uptime SLA
- Handle files up to 2GB

## 🔬 EEG Processing Standards

### EEGPT Specifications

- **Sampling**: 256 Hz (resample if needed)
- **Windows**: 4 seconds (1024 samples) for TUAB linear probe
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
# PARALLEL PATHWAYS (not sequential!):

# Path 1: EEGPT Pipeline (requires 19+ channels for clinical use)
Raw EEG (256Hz) → Autoreject (QC) → EEGPT (Features) → Task Head (Prediction)

# Path 2: YASA Pipeline (works with ANY channel count, 1-100+)
Any EEG → Channel Selection (prefers C3/C4) → YASA → Sleep Stages

# KEY FACTS:
# - YASA is NOT limited to 2 channels - works with any count
# - YASA achieves 85%+ accuracy with just 1 central EEG channel
# - Sleep-EDF has 2 channels but that's dataset-specific, not a YASA requirement

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

Implement in this order:

### 1. Basic EEG Pipeline

```bash
# Test basic model loading and inference
uv run python scripts/testing/test_sleep_analysis.py

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

### 🚨 BEFORE ADDING ANYTHING - THE CHECKLIST

```bash
# 1. CHECK IF IT EXISTS
grep -r "class.*YourThing" src/
find src/ -name "*your_feature*"

# 2. IF IT EXISTS, USE IT
from brain_go_brrr.existing.module import ExistingThing

# 3. IF IT DOESN'T EXIST, ADD TO src/ NOT experiments/
# Create in: src/brain_go_brrr/appropriate/location/

# 4. THEN USE FROM experiments/
from brain_go_brrr.appropriate.location import NewThing
```

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
Reference domain/quality/controller.py for structure"

# 4. Verify and document
make test
make docs
```

### Processing Sleep-EDF Data

```python
from pathlib import Path
import mne
from services.sleep_metrics import SleepAnalyzer

# Load data using config
from brain_go_brrr.application.config import DataConfig
config = DataConfig()
edf_path = config.get_sleep_edf_psg_file()  # Gets first PSG file deterministically
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
uv run python -m memory_profiler scripts/testing/test_sleep_analysis.py

# Check Sleep-EDF data
find data/datasets/sleep-edf -name "*.edf" | wc -l
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

## ❌ Do NOT (ABSOLUTE FORBIDDEN ACTIONS)

### 🔥 ARCHITECTURE VIOLATIONS (NEVER DO THESE)
- **CREATE PARALLEL IMPLEMENTATIONS** in experiments/ and src/
- **REIMPLEMENT** existing src/ components in experiments/
- **BUILD IN ISOLATION** without checking src/ first
- **DUPLICATE** datasets, models, or preprocessing
- **IGNORE** existing working code
- **CREATE** new files without grep searching first

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
