# Brain-Go-Brrr Architecture

## Overview

Brain-Go-Brrr is a production-ready EEG analysis system using Clean Architecture principles with frozen EEGPT features and specialized task heads.

## System Architecture - Parallel Processing Pathways

```
┌─────────────────────────────────────────────────────────────┐
│                        EEG Input                            │
│                (.edf/.bdf files - ANY channel count)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Quality Control (QC)                      │
│              AutoReject + Bad Channel Detection             │
│                    Status: ✅ WORKING                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
   EEGPT Pipeline                  YASA Pipeline
  (Requires 19+ channels)         (Works with ANY count)
  (256Hz or resample)             (Auto-selects best channel)
        │                               │
        ▼                               ▼
┌──────────────────────┐    ┌──────────────────────┐
│  EEGPT Features      │    │   YASA Sleep Staging │
│  512-dim embeddings  │    │  85%+ accuracy w/ 1ch│
│  Status: ✅ WORKING  │    │   Status: ✅ WORKING │
└──────┬───────────────┘    └──────────────────────┘
       │                                │
       ├──────────────┐                 │
       │              │                 ▼
       ▼              ▼           Sleep Metrics
┌──────────────┐  ┌──────────────┐  & Hypnogram
│ Abnormality  │  │ Sleep Probe  │
│ Detection    │  │ (EEGPT-based)│
│ Status: 🟡   │  │ Status: 🔬   │
└──────────────┘  └──────────────┘

IMPORTANT: EEGPT and YASA are PARALLEL pathways, not sequential.
- YASA works with ANY channel count (1-100+), not just 2
- YASA auto-selects the best central channel (C3/C4 preferred)
- Sleep-EDF has 2 channels but that's dataset-specific, not a YASA limitation
- Full EEG (256Hz, 19+ch) → EEGPT → Linear Probes
```

## Clean Architecture Layers

```
src/brain_go_brrr/
├── domain/                 # Business logic (no external dependencies)
│   ├── quality/           # QC domain models and ports
│   ├── sleep/            # Sleep analysis domain
│   └── exceptions.py     # Domain-specific exceptions
│
├── application/           # Use case orchestration
│   ├── pipeline/         # Analysis pipelines
│   └── factories.py      # Dependency injection
│
├── infra/                 # External integrations
│   ├── ml_models/        # EEGPT model implementation
│   │   ├── eegpt_compat.py      # Model compatibility layer
│   │   └── eegpt_architecture.py # Architecture definition
│   └── external/         # Third-party adapters
│       ├── yasa_adapter.py      # YASA sleep staging
│       └── autoreject_adapter.py # Autoreject QC
│
└── api/                   # Presentation layer
    ├── main.py           # FastAPI application
    ├── routers/          # API endpoints
    └── schemas.py        # Request/response models
```

## Component Details

### 1. Quality Control (Autoreject) ✅

**Purpose**: Detect and handle bad channels and artifacts before analysis.

**Implementation**:
- Bad channel detection with 87% expert agreement
- Epoch-level artifact rejection
- Automated interpolation of bad channels
- Chunked processing for memory efficiency

**Key Interface**:
```python
class QualityController:
    def run_full_qc_pipeline(self, raw: mne.io.Raw) -> QCReport:
        """Run complete quality control pipeline."""
```

### 2. EEGPT Feature Extraction ✅

**Purpose**: Extract rich representations from EEG using pretrained transformer.

**Model Specs**:
- 10M parameters, trained on TUH EEG
- 4-second windows at 256Hz
- 512-dimensional embeddings per patch
- 4 summary tokens per window

**Key Interface**:
```python
class EEGPTModel:
    def extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract EEGPT features from EEG data.
        Returns: (n_windows, 4, 512) features
        """
```

### 3. Sleep Analysis (YASA) ✅

**Purpose**: Classify sleep stages from overnight EEG recordings.

**Implementation**:
- 5-stage classification (W, N1, N2, N3, REM)
- 85%+ accuracy with just 1 central EEG channel
- 87% accuracy with 1 EEG + 1 EOG channel
- Works with ANY channel count (auto-selects best)
- Automatic channel aliasing for non-standard montages
- Hypnogram and sleep metrics generation

**Key Facts**:
- **NOT limited to 2 channels** - works with 1 to 100+ channels
- Prefers central channels (C3/C4) for best accuracy
- Sleep-EDF happens to have 2 channels, but that's dataset-specific

**Channel Aliasing** (improves accuracy for non-standard montages):
```python
"EEG Fpz-Cz" → "C4"  # Frontal to Central (Sleep-EDF)
"EEG Pz-Oz" → "O2"   # Parietal to Occipital (Sleep-EDF)
"Fpz" → "C3"         # Single electrode mapping
```

### 4. Abnormality Detection 🟡 (Training)

**Purpose**: Binary classification of normal vs abnormal EEG patterns.

**Implementation**:
- Linear probe on frozen EEGPT features
- Training on TUAB dataset (~1.86M windows)
- Target: 0.87 AUROC (paper performance)
- Current: Training at batch 292/7286

**Training Location**: `experiments/eegpt_linear_probe/`

### 5. API Layer ✅

**Purpose**: RESTful API for EEG analysis services.

**Endpoints**:
- `GET /health` - Service health check
- `POST /api/v1/eeg/analyze` - Run analysis pipeline
- `GET /api/v1/eeg/results/{job_id}` - Get analysis results

**Features**:
- Redis caching for performance
- Async processing support
- OpenAPI documentation
- CORS enabled for web clients

## Data Flow

1. **Input**: EDF/BDF file upload or S3 reference
2. **Preprocessing**: Resample to 256Hz, bandpass filter (0.5-50Hz)
3. **Quality Control**: Autoreject pipeline for artifact handling
4. **Feature Extraction**: EEGPT embeddings (frozen backbone)
5. **Task Heads**: Parallel analysis (sleep, abnormality, etc.)
6. **Output**: JSON results with confidence scores

## Performance Characteristics

| Component | Processing Time | Memory Usage |
|-----------|----------------|--------------|
| QC Pipeline | <30s per hour | ~2GB |
| EEGPT Features | ~100ms per window | ~500MB model |
| Sleep Staging | <30s for 8 hours | ~1GB |
| API Response | <100ms | Cached |

## Design Principles

1. **Clean Architecture**: Domain logic independent of frameworks
2. **Dependency Injection**: Flexible component composition
3. **Adapter Pattern**: Third-party libraries behind interfaces
4. **Repository Pattern**: Data access abstraction
5. **SOLID Principles**: Single responsibility, open/closed, etc.

## Future Components (Not Implemented)

- **Event Detection**: Seizure and IED detection
- **Authentication**: OAuth2 with JWT tokens
- **Message Queue**: Celery for async processing
- **Database**: PostgreSQL with TimescaleDB
- **Deployment**: Kubernetes orchestration

## Testing Strategy

- **Unit Tests**: 454 passing tests
- **Integration Tests**: Real EEG data validation
- **Performance Tests**: Benchmark suite
- **CI/CD**: GitHub Actions (all branches green)
