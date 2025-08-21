# Architecture Review - Brain-Go-Brrr
*Date: August 12, 2025*
*Branch: feature/architecture-refactor*

## Executive Summary
After comprehensive analysis of the codebase structure, I've identified several architectural issues that violate SOLID principles and Clean Architecture patterns. While the codebase is functional with 694 passing tests and 64.79% coverage, there are opportunities to improve maintainability, reduce coupling, and enhance clarity.

## Current Architecture Analysis

### Directory Structure Overview
```
src/brain_go_brrr/
├── api/          (18 files) - REST API layer
├── core/         (26 files) - Business logic (BLOATED)
├── data/         (5 files)  - Data loading/datasets
├── models/       (8 files)  - ML models
├── preprocessing/(6 files)  - EEG preprocessing
├── services/     (4 files)  - Service layer (has duplicates)
├── tasks/        (3 files)  - Background tasks
├── training/     (1 file)   - Training logic
├── infra/        (6 files)  - Infrastructure
├── inference/    (1 file)   - NEARLY EMPTY
├── modules/      (2 files)  - Constraints/utilities
├── utils/        (2 files)  - General utilities
└── visualization/(3 files)  - Reporting
```

## 🔴 Critical Issues Found

### 1. **Single Responsibility Principle (SRP) Violations**

#### Core Module Bloat (26 files)
The `core/` module is doing too much:
- Configuration management (`config.py`, `abnormality_config.py`)
- Data loading (`edf_loader.py`, `edf_validator.py`)
- Preprocessing (`preprocessing.py`)
- Feature extraction (`window_extractor.py`)
- Multiple domain concepts (abnormal/, sleep/, quality/, snippets/)

**Impact**: High coupling, difficult to test in isolation, unclear boundaries

#### Duplicate Implementations
- **YASA Adapters**: 3 versions in services/
  - `yasa_adapter.py`
  - `yasa_adapter_enhanced.py`
  - `yasa_adapter_original_backup.py` (❌ backup files in production!)

### 2. **Open/Closed Principle (OCP) Violations**

#### Preprocessing Scattered Across Modules
- `core/preprocessing.py` - Core preprocessing
- `preprocessing/` directory - Specialized preprocessing
- No clear abstraction or interface

**Impact**: Adding new preprocessing requires modifying multiple locations

### 3. **Dependency Inversion Principle (DIP) Violations**

#### Direct Coupling to External Libraries
- Models directly import torch/lightning
- Services directly coupled to YASA/MNE
- No abstraction layer for external dependencies

**Impact**: Hard to swap implementations, difficult to mock for testing

### 4. **Interface Segregation Principle (ISP) Violations**

#### Fat Interfaces in Core
- `core/` subdirectories mixing concerns:
  - `abnormal/` - detection logic
  - `sleep/` - analysis logic
  - `quality/` - QC logic
  - `features/` - extraction logic

**Impact**: Clients forced to depend on interfaces they don't use

### 5. **Organizational Drift**

#### Misplaced Files
- `mne_compat.py` at package root (should be in `infra/` or `utils/`)
- `_typing.py` at package root (should be in `utils/` or types module)
- Empty/underutilized modules (`inference/`, `config/`)

#### Unclear Module Boundaries
- Overlap between `core/`, `preprocessing/`, and `data/`
- Confusion between `tasks/` and `services/`
- `modules/` vs `utils/` distinction unclear

## 🟡 Code Smells Detected

### 1. **Feature Envy**
- `core/edf_loader.py` and `core/edf_validator.py` should be in `data/`
- `core/window_extractor.py` belongs in `preprocessing/` or `features/`

### 2. **Shotgun Surgery Pattern**
- Channel handling spread across:
  - `core/channels.py`
  - Various preprocessing modules
  - Individual model files

### 3. **Parallel Inheritance Hierarchies**
- Multiple EEGPT model variants:
  - `eegpt_model.py`
  - `eegpt_linear_probe.py`
  - `eegpt_linear_probe_robust.py`
  - `eegpt_two_layer_probe.py`
  - `eegpt_wrapper.py`

### 4. **Dead Code**
- `yasa_adapter_original_backup.py` (backup file!)
- Nearly empty `inference/` module
- Underutilized `config/` module

## 🟢 What's Working Well

### Positive Patterns
1. **Clear API layer separation** - FastAPI routers well organized
2. **Type hints throughout** - Good typing discipline
3. **Test organization** - Clear unit/integration/benchmark separation
4. **Infrastructure abstraction** - Redis/cache properly isolated

### Should Keep
- Current test structure
- API router organization
- Type hinting approach
- Documentation structure

## Recommended Architecture

### Proposed Clean Architecture Layers

```
src/brain_go_brrr/
├── domain/           # Core business entities & rules
│   ├── entities/     # EEG, Patient, Study, etc.
│   ├── value_objects/# Channels, TimeWindow, etc.
│   └── exceptions/   # Domain exceptions
│
├── application/      # Use cases & business logic
│   ├── abnormality/  # Abnormality detection use cases
│   ├── sleep/        # Sleep analysis use cases
│   ├── quality/      # Quality control use cases
│   └── interfaces/   # Port interfaces
│
├── infrastructure/   # External dependencies
│   ├── data/         # EDF loading, datasets
│   ├── models/       # ML model implementations
│   ├── external/     # MNE, YASA, etc. adapters
│   ├── persistence/  # Redis, cache, storage
│   └── config/       # Configuration management
│
├── presentation/     # UI/API layer
│   ├── api/          # FastAPI routers
│   ├── cli/          # CLI interface
│   └── schemas/      # Request/response schemas
│
└── shared/           # Cross-cutting concerns
    ├── preprocessing/# Preprocessing pipelines
    ├── features/     # Feature extraction
    └── utils/        # General utilities
```

## Design Pattern Recommendations

### 1. **Strategy Pattern** for Preprocessing
Replace multiple preprocessing classes with strategy pattern:
```python
class PreprocessingStrategy(ABC):
    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray: ...

class AutorejectStrategy(PreprocessingStrategy): ...
class BandpassStrategy(PreprocessingStrategy): ...
```

### 2. **Factory Pattern** for Model Creation
Consolidate EEGPT variants:
```python
class EEGPTFactory:
    @staticmethod
    def create(model_type: ModelType) -> BaseEEGPT: ...
```

### 3. **Repository Pattern** for Data Access
Abstract data loading:
```python
class EEGRepository(ABC):
    @abstractmethod
    def load(self, path: Path) -> EEGData: ...

class EDFRepository(EEGRepository): ...
class TUABRepository(EEGRepository): ...
```

### 4. **Adapter Pattern** for External Libraries
Wrap external dependencies:
```python
class SleepAnalyzer(ABC):
    @abstractmethod
    def analyze(self, data: EEGData) -> SleepMetrics: ...

class YASAAdapter(SleepAnalyzer): ...
```

## Quick Wins (Can Do Now)

1. **Remove backup files**: Delete `yasa_adapter_original_backup.py`
2. **Move misplaced files**:
   - `core/edf_loader.py` → `data/`
   - `core/edf_validator.py` → `data/`
   - `core/window_extractor.py` → `preprocessing/`
   - `mne_compat.py` → `infrastructure/external/`
3. **Consolidate empty modules**: Remove or repurpose `inference/`, `config/`
4. **Create interfaces module**: Define clear contracts
5. **Deduplicate YASA adapters**: Keep only one implementation

## Risk Assessment

### Low Risk Refactors
- Moving files to correct modules
- Removing dead code
- Creating interface definitions
- Consolidating duplicate code

### Medium Risk Refactors
- Restructuring core module
- Implementing repository pattern
- Adding abstraction layers

### High Risk Refactors
- Complete restructure to clean architecture
- Changing public API contracts
- Modifying model loading logic

## Metrics for Success

- **Reduce core module** from 26 to <10 files
- **Increase cohesion**: Each module single purpose
- **Reduce coupling**: Dependency injection throughout
- **Improve testability**: Mock external dependencies easily
- **Maintain coverage**: Keep at 65%+ during refactor

## Conclusion

The codebase shows signs of organic growth without architectural governance. While functional, it violates several SOLID principles and contains organizational drift. The proposed refactoring will:

1. Improve maintainability through clear boundaries
2. Enhance testability through dependency injection
3. Reduce coupling through abstraction layers
4. Increase cohesion through proper module organization
5. Follow Clean Architecture and SOLID principles

The refactoring can be done incrementally, starting with quick wins and gradually moving to larger structural changes.
