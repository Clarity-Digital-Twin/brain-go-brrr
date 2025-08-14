# 🏛️ CLEAN ARCHITECTURE IMPLEMENTATION

## Uncle Bob's Clean Architecture - FULLY IMPLEMENTED

This codebase follows Robert C. Martin's Clean Architecture principles to the letter. Every architectural decision enforces the Dependency Rule: **source code dependencies point only inward toward higher-level policies**.

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRESENTATION                            │
│                    (API Endpoints, CLI, UI)                     │
├─────────────────────────────────────────────────────────────────┤
│                         APPLICATION                             │
│              (Use Cases, Factories, Orchestration)             │
├─────────────────────────────────────────────────────────────────┤
│                           DOMAIN                                │
│            (Entities, Value Objects, Domain Services)           │
│                     ZERO DEPENDENCIES                           │
└─────────────────────────────────────────────────────────────────┘
        ↑                       ↑                        ↑
        │                       │                        │
┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ INFRASTRUCTURE│    │   INFRASTRUCTURE  │    │  INFRASTRUCTURE  │
│   (Database)  │    │   (File System)   │    │    (External)    │
└───────────────┘    └──────────────────┘    └──────────────────┘
```

## 🎯 Core Principles

### 1. **Dependency Rule**
- Dependencies point **ONLY INWARD**
- Domain has **ZERO** dependencies on outer layers
- Outer layers depend on inner layers through **abstractions**

### 2. **Dependency Inversion Principle (DIP)**
- High-level modules don't depend on low-level modules
- Both depend on **abstractions** (ports/interfaces)
- Abstractions don't depend on details
- Details depend on abstractions

### 3. **Separation of Concerns**
- **Domain**: Business logic and rules
- **Application**: Use case orchestration
- **Infrastructure**: External concerns (DB, files, APIs)
- **Presentation**: User interface (API, CLI, Web)

## 📁 Project Structure

```
src/brain_go_brrr/
├── domain/                 # PURE BUSINESS LOGIC (Zero dependencies)
│   ├── abnormal/
│   │   ├── ports.py       # Domain-defined interfaces
│   │   ├── settings.py    # Pure value objects
│   │   ├── detector_pure.py # Pure domain service
│   │   └── detector.py    # Legacy (being migrated)
│   ├── quality/
│   │   ├── ports.py       # Quality control interfaces
│   │   └── controller_clean.py
│   └── preprocessing/
│       └── features/
│           └── extractor_clean.py
│
├── application/            # USE CASE ORCHESTRATION
│   ├── factories_pure.py  # Composition root (wires everything)
│   ├── factories.py       # Legacy factories
│   └── config.py          # Application configuration
│
├── infra/                  # INFRASTRUCTURE IMPLEMENTATIONS
│   ├── adapters/          # Implement domain ports
│   │   ├── preprocessor_flexible.py
│   │   ├── eegpt_classifier.py
│   │   ├── eegpt_feature_extractor.py
│   │   ├── logger_adapter.py
│   │   └── autoreject_adapter.py
│   ├── ml_models/         # Concrete ML implementations
│   ├── preprocessing/     # Concrete preprocessing
│   └── data/             # Data access layer
│
└── api/                    # PRESENTATION LAYER
    ├── routers/           # FastAPI endpoints
    └── schemas.py         # API contracts
```

## 🔌 Ports and Adapters Pattern

### Domain Ports (Interfaces)

```python
# domain/abnormal/ports.py
@runtime_checkable
class EEGPreprocessorPort(Protocol):
    """Domain defines what it needs."""
    def transform(self, raw: MneRaw) -> npt.NDArray[np.float32]: ...

@runtime_checkable
class AbnormalityHeadPort(Protocol):
    """Domain defines the contract."""
    def predict_proba(self, X: npt.NDArray[np.float32]) -> float: ...
```

### Infrastructure Adapters (Implementations)

```python
# infra/adapters/preprocessor_flexible.py
class FlexiblePreprocessorAdapter(EEGPreprocessorPort):
    """Infrastructure implements the contract."""
    def __init__(self, **kwargs):
        self._inner = FlexibleEEGPreprocessor(**kwargs)
    
    def transform(self, raw: MneRaw) -> npt.NDArray[np.float32]:
        # Adapter wraps infrastructure implementation
        return self._inner.preprocess(raw).astype("float32")
```

### Application Wiring (Composition Root)

```python
# application/factories_pure.py
def create_pure_abnormality_detector(...) -> PureAbnormalityDetector:
    """The ONLY place that knows about concrete implementations."""
    
    # Create infrastructure components
    preprocessor = FlexiblePreprocessorAdapter(...)
    classifier = EEGPTClassifierAdapter(...)
    
    # Wire into domain service
    detector = PureAbnormalityDetector(
        preprocessor=preprocessor,  # Port satisfied by adapter
        classifier=classifier,      # Port satisfied by adapter
        settings=settings,          # Pure value object
        logger=logger               # Port satisfied by adapter
    )
    return detector
```

## 🧪 Testing Strategy

### 1. **Domain Unit Tests** (FAST & PURE)
```python
def test_pure_domain_logic():
    # Create stub implementations
    preprocessor = StubPreprocessor()
    classifier = StubClassifier(returns=0.8)
    
    # Test pure domain logic
    detector = PureAbnormalityDetector(
        preprocessor=preprocessor,
        classifier=classifier,
        settings=AbnormalitySettings(),
    )
    
    result = detector.detect(mock_raw)
    assert result.is_abnormal == True
```

### 2. **Integration Tests** (WITH REAL ADAPTERS)
```python
def test_with_real_infrastructure():
    # Use factory to wire real implementations
    detector = create_pure_abnormality_detector(
        config=test_config,
        model_path=test_model_path,
    )
    
    result = detector.detect(real_eeg_data)
    assert result.triage_level == TriageLevel.URGENT
```

### 3. **Contract Tests** (VERIFY ADAPTERS)
```python
def test_adapter_implements_port():
    adapter = FlexiblePreprocessorAdapter()
    # Verify it implements the protocol
    assert isinstance(adapter, EEGPreprocessorPort)
```

## 🚔 Enforcement with Import-Linter

### Strict Contracts
```ini
[importlinter:contract:domain_pure]
name = Domain must be PURE
type = forbidden
source_modules = brain_go_brrr.domain
forbidden_modules = 
    brain_go_brrr.application
    brain_go_brrr.infra
    brain_go_brrr.api
```

### Validation Commands
```bash
# Check architecture compliance
make import-lint

# Verify no domain leaks
git grep -n "from brain_go_brrr\.infra" src/brain_go_brrr/domain

# Run all gates
make lint && make type-check && make test && make import-lint
```

## 🎓 Clean Architecture Benefits

### 1. **Testability**
- Domain logic testable without infrastructure
- Fast unit tests with stubs
- No database/network needed for domain tests

### 2. **Flexibility**
- Swap implementations without changing domain
- Multiple adapters for same port
- Easy to add new infrastructure

### 3. **Maintainability**
- Business logic isolated and pure
- Clear boundaries between layers
- Dependencies always explicit

### 4. **Independence**
- Domain doesn't know about frameworks
- Domain doesn't know about databases
- Domain doesn't know about external services

## 📚 Examples

### Creating a Detector
```python
from brain_go_brrr.application.factories_pure import (
    create_pure_abnormality_detector
)

# Application layer creates and wires
detector = create_pure_abnormality_detector(
    config=my_config,
    model_path="path/to/model.ckpt",
    device="cuda",
)

# Use the pure domain service
result = detector.detect(eeg_data)
print(f"Abnormal: {result.is_abnormal}")
print(f"Triage: {result.triage_level}")
```

### Adding a New Infrastructure Implementation
```python
# 1. Domain already defines the port
# domain/abnormal/ports.py - NO CHANGES NEEDED

# 2. Create new adapter
class CloudMLAdapter(AbnormalityHeadPort):
    def predict_proba(self, X):
        # Call cloud API
        response = requests.post(...)
        return response.json()["probability"]

# 3. Wire in factory
def create_cloud_detector():
    classifier = CloudMLAdapter(api_key=...)
    # Rest stays the same!
```

## 🚀 Migration Status

### ✅ Completed
- Domain ports defined
- Pure domain services created
- Infrastructure adapters implemented
- Application factories with DI
- Import-linter strict enforcement

### 🔄 In Progress
- Migrating legacy detector.py to detector_pure.py
- Updating all tests to use factories
- Documentation updates

### 📋 TODO
- Complete migration of all domain services
- Add contract tests for all adapters
- Performance benchmarks for pure vs legacy

## 📖 References

- [Clean Architecture by Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Dependency Inversion Principle](https://en.wikipedia.org/wiki/Dependency_inversion_principle)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Ports and Adapters Pattern](https://en.wikipedia.org/wiki/Hexagonal_architecture_(software))

---

*"The center of your application is not the database. Nor is it one or more of the frameworks you may be using. **The center of your application is the use cases of your application**."* - Robert C. Martin