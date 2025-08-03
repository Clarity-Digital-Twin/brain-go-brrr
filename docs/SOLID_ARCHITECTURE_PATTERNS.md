# SOLID Architecture Patterns for Brain-Go-Brrr

## Executive Summary

This document defines SOLID principles implementation for the Brain-Go-Brrr EEG analysis pipeline. By following these patterns, we ensure maintainable, extensible, and testable code that can evolve with changing requirements while maintaining stability.

## SOLID Principles Overview

1. **S**ingle Responsibility Principle (SRP)
2. **O**pen/Closed Principle (OCP)
3. **L**iskov Substitution Principle (LSP)
4. **I**nterface Segregation Principle (ISP)
5. **D**ependency Inversion Principle (DIP)

## 1. Single Responsibility Principle (SRP)

### Definition
A class should have only one reason to change, meaning it should have only one job or responsibility.

### Implementation Examples

#### ✅ Good: Separated Concerns
```python
# src/brain_go_brrr/models/eegpt_model.py
class EEGPTModel:
    """Responsible ONLY for EEGPT model operations."""
    
    def __init__(self, checkpoint_path: Path):
        self.model = self._load_checkpoint(checkpoint_path)
    
    def extract_features(self, eeg_window: np.ndarray) -> torch.Tensor:
        """Extract features from EEG window."""
        return self.model.encode(eeg_window)

# src/brain_go_brrr/data/preprocessor.py
class EEGPreprocessor:
    """Responsible ONLY for EEG preprocessing."""
    
    def __init__(self, target_sfreq: int = 256):
        self.target_sfreq = target_sfreq
    
    def preprocess(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply preprocessing pipeline."""
        raw = self._resample(raw)
        raw = self._filter(raw)
        raw = self._normalize(raw)
        return raw

# src/brain_go_brrr/data/channel_mapper.py
class ChannelMapper:
    """Responsible ONLY for channel name mapping."""
    
    OLD_TO_MODERN = {
        "T3": "T7", "T4": "T8", 
        "T5": "P7", "T6": "P8"
    }
    
    def map_channels(self, channels: List[str]) -> List[str]:
        """Map old naming to modern convention."""
        return [self.OLD_TO_MODERN.get(ch, ch) for ch in channels]
```

#### ❌ Bad: Multiple Responsibilities
```python
class EEGPTProcessor:
    """Violates SRP - does too many things."""
    
    def process(self, edf_path: str) -> Dict:
        # File loading responsibility
        raw = mne.io.read_raw_edf(edf_path)
        
        # Preprocessing responsibility
        raw.resample(256)
        raw.filter(0.5, 50)
        
        # Channel mapping responsibility
        if "T3" in raw.ch_names:
            raw.rename_channels({"T3": "T7"})
        
        # Model inference responsibility
        model = torch.load("model.ckpt")
        features = model(raw.get_data())
        
        # Results formatting responsibility
        return {"features": features.tolist()}
```

### SRP in Service Layer

```python
# src/brain_go_brrr/services/quality_control.py
class QualityControlService:
    """Single responsibility: Coordinate quality control workflow."""
    
    def __init__(
        self,
        autoreject: AutoRejectWrapper,
        channel_validator: ChannelValidator,
        artifact_detector: ArtifactDetector
    ):
        self.autoreject = autoreject
        self.channel_validator = channel_validator
        self.artifact_detector = artifact_detector
    
    async def assess_quality(self, raw: mne.io.Raw) -> QualityReport:
        """Orchestrate quality assessment."""
        # Delegate to specialized components
        channel_status = await self.channel_validator.validate(raw)
        artifacts = await self.artifact_detector.detect(raw)
        cleaned = await self.autoreject.process(raw)
        
        return QualityReport(
            channel_status=channel_status,
            artifacts=artifacts,
            cleaned_data=cleaned
        )
```

## 2. Open/Closed Principle (OCP)

### Definition
Software entities should be open for extension but closed for modification.

### Implementation Examples

#### ✅ Good: Extensible via Abstraction
```python
# src/brain_go_brrr/models/base.py
from abc import ABC, abstractmethod

class BaseEEGModel(ABC):
    """Abstract base for all EEG models."""
    
    @abstractmethod
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """Extract features from EEG data."""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return feature dimension."""
        pass

# src/brain_go_brrr/models/eegpt.py
class EEGPTModel(BaseEEGModel):
    """EEGPT implementation."""
    
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        # EEGPT-specific implementation
        return self.transformer.encode(eeg_data)
    
    def get_feature_dim(self) -> int:
        return 768

# src/brain_go_brrr/models/bendr.py
class BENDRModel(BaseEEGModel):
    """BENDR implementation - can be added without modifying existing code."""
    
    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        # BENDR-specific implementation
        return self.cnn_transformer.encode(eeg_data)
    
    def get_feature_dim(self) -> int:
        return 512
```

#### Strategy Pattern for Extensibility
```python
# src/brain_go_brrr/detection/strategies.py
from abc import ABC, abstractmethod

class DetectionStrategy(ABC):
    """Base strategy for event detection."""
    
    @abstractmethod
    def detect(self, eeg_window: np.ndarray) -> List[Event]:
        pass

class EpileptiformStrategy(DetectionStrategy):
    """Detect epileptiform discharges."""
    
    def detect(self, eeg_window: np.ndarray) -> List[Event]:
        # Spike detection algorithm
        peaks = self._find_sharp_peaks(eeg_window)
        return [Event("epileptiform", p.time, p.confidence) for p in peaks]

class PLEDStrategy(DetectionStrategy):
    """Detect periodic lateralized epileptiform discharges."""
    
    def detect(self, eeg_window: np.ndarray) -> List[Event]:
        # Periodic pattern detection
        patterns = self._find_periodic_patterns(eeg_window)
        return [Event("PLED", p.start, p.frequency) for p in patterns]

# Usage - easily extensible
class EventDetector:
    """Detector using strategy pattern."""
    
    def __init__(self, strategy: DetectionStrategy):
        self.strategy = strategy
    
    def detect_events(self, eeg_data: np.ndarray) -> List[Event]:
        return self.strategy.detect(eeg_data)
```

### Plugin Architecture
```python
# src/brain_go_brrr/plugins/base.py
class AnalysisPlugin(ABC):
    """Base class for analysis plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @abstractmethod
    def analyze(self, features: np.ndarray) -> AnalysisResult:
        """Perform analysis on features."""
        pass

# src/brain_go_brrr/plugins/registry.py
class PluginRegistry:
    """Registry for dynamic plugin loading."""
    
    def __init__(self):
        self._plugins: Dict[str, Type[AnalysisPlugin]] = {}
    
    def register(self, plugin_class: Type[AnalysisPlugin]):
        """Register a new plugin."""
        self._plugins[plugin_class.name] = plugin_class
    
    def get_plugin(self, name: str) -> AnalysisPlugin:
        """Get plugin instance by name."""
        return self._plugins[name]()

# New plugins can be added without modifying core
@registry.register
class SeizureRiskPlugin(AnalysisPlugin):
    """Custom seizure risk assessment plugin."""
    
    name = "seizure_risk"
    
    def analyze(self, features: np.ndarray) -> AnalysisResult:
        # Custom analysis logic
        risk_score = self._calculate_seizure_risk(features)
        return AnalysisResult(score=risk_score, type="seizure_risk")
```

## 3. Liskov Substitution Principle (LSP)

### Definition
Objects of a superclass should be replaceable with objects of its subclasses without breaking the application.

### Implementation Examples

#### ✅ Good: Proper Inheritance
```python
# src/brain_go_brrr/data/loaders.py
class BaseDataLoader(ABC):
    """Base class for all data loaders."""
    
    @abstractmethod
    def load(self, path: Path) -> mne.io.Raw:
        """Load EEG data from path."""
        pass
    
    @abstractmethod
    def validate(self, raw: mne.io.Raw) -> bool:
        """Validate loaded data."""
        pass

class EDFLoader(BaseDataLoader):
    """EDF file loader - maintains LSP."""
    
    def load(self, path: Path) -> mne.io.Raw:
        """Load EDF file."""
        if not path.suffix == '.edf':
            raise ValueError(f"Expected .edf file, got {path.suffix}")
        return mne.io.read_raw_edf(path, preload=False)
    
    def validate(self, raw: mne.io.Raw) -> bool:
        """Validate EDF data."""
        return len(raw.ch_names) >= 19 and raw.info['sfreq'] > 0

class FIFLoader(BaseDataLoader):
    """FIF file loader - maintains LSP."""
    
    def load(self, path: Path) -> mne.io.Raw:
        """Load FIF file."""
        if not path.suffix == '.fif':
            raise ValueError(f"Expected .fif file, got {path.suffix}")
        return mne.io.read_raw_fif(path, preload=False)
    
    def validate(self, raw: mne.io.Raw) -> bool:
        """Validate FIF data."""
        return len(raw.ch_names) >= 19 and raw.info['sfreq'] > 0

# LSP in action - can use any loader interchangeably
def process_eeg_file(loader: BaseDataLoader, path: Path) -> ProcessedData:
    """Works with any loader implementation."""
    raw = loader.load(path)
    if not loader.validate(raw):
        raise ValueError("Invalid EEG data")
    return process_raw_data(raw)
```

#### ❌ Bad: LSP Violation
```python
class Bird:
    def fly(self):
        return "Flying"

class Penguin(Bird):
    def fly(self):
        # LSP violation - penguins can't fly!
        raise NotImplementedError("Penguins can't fly")
```

### Proper Abstraction Hierarchies
```python
# src/brain_go_brrr/models/classifiers.py
class BaseClassifier(ABC):
    """Base classifier with common interface."""
    
    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probability for each class."""
        pass
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return predicted class."""
        proba = self.predict_proba(features)
        return np.argmax(proba, axis=1)

class AbnormalityClassifier(BaseClassifier):
    """Binary classifier for normal/abnormal."""
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return [P(normal), P(abnormal)]."""
        logits = self.model(features)
        return torch.softmax(logits, dim=1).numpy()

class SleepStageClassifier(BaseClassifier):
    """Multi-class sleep stage classifier."""
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return probabilities for W, N1, N2, N3, REM."""
        logits = self.model(features)
        return torch.softmax(logits, dim=1).numpy()

# Both classifiers can be used interchangeably
def evaluate_classifier(
    classifier: BaseClassifier, 
    test_features: np.ndarray,
    test_labels: np.ndarray
) -> float:
    """Works with any classifier following LSP."""
    predictions = classifier.predict(test_features)
    return accuracy_score(test_labels, predictions)
```

## 4. Interface Segregation Principle (ISP)

### Definition
Clients should not be forced to depend on interfaces they don't use.

### Implementation Examples

#### ✅ Good: Segregated Interfaces
```python
# src/brain_go_brrr/interfaces/analysis.py
from abc import ABC, abstractmethod

class QualityAnalyzer(ABC):
    """Interface for quality analysis."""
    
    @abstractmethod
    def analyze_quality(self, raw: mne.io.Raw) -> QualityReport:
        pass

class AbnormalityDetector(ABC):
    """Interface for abnormality detection."""
    
    @abstractmethod
    def detect_abnormality(self, features: np.ndarray) -> AbnormalityResult:
        pass

class SleepAnalyzer(ABC):
    """Interface for sleep analysis."""
    
    @abstractmethod
    def analyze_sleep(self, raw: mne.io.Raw) -> SleepReport:
        pass

class EventDetector(ABC):
    """Interface for event detection."""
    
    @abstractmethod
    def detect_events(self, raw: mne.io.Raw) -> List[Event]:
        pass

# Implementations only implement what they need
class BasicQualityChecker(QualityAnalyzer):
    """Only implements quality checking."""
    
    def analyze_quality(self, raw: mne.io.Raw) -> QualityReport:
        # Quality analysis implementation
        return QualityReport(...)

class ComprehensiveAnalyzer(
    QualityAnalyzer, 
    AbnormalityDetector,
    SleepAnalyzer
):
    """Implements multiple interfaces as needed."""
    
    def analyze_quality(self, raw: mne.io.Raw) -> QualityReport:
        # Implementation
        pass
    
    def detect_abnormality(self, features: np.ndarray) -> AbnormalityResult:
        # Implementation
        pass
    
    def analyze_sleep(self, raw: mne.io.Raw) -> SleepReport:
        # Implementation
        pass
```

#### ❌ Bad: Fat Interface
```python
class EEGAnalyzer(ABC):
    """Fat interface - forces all implementations to implement everything."""
    
    @abstractmethod
    def analyze_quality(self, raw: mne.io.Raw) -> QualityReport:
        pass
    
    @abstractmethod
    def detect_abnormality(self, features: np.ndarray) -> AbnormalityResult:
        pass
    
    @abstractmethod
    def analyze_sleep(self, raw: mne.io.Raw) -> SleepReport:
        pass
    
    @abstractmethod
    def detect_events(self, raw: mne.io.Raw) -> List[Event]:
        pass
    
    @abstractmethod
    def calculate_complexity(self, raw: mne.io.Raw) -> float:
        pass
    
    # Forces simple quality checker to implement unused methods
```

### Role-Based Interfaces
```python
# src/brain_go_brrr/interfaces/storage.py
class Readable(ABC):
    """Interface for reading data."""
    
    @abstractmethod
    def read(self, key: str) -> bytes:
        pass

class Writable(ABC):
    """Interface for writing data."""
    
    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        pass

class Deletable(ABC):
    """Interface for deleting data."""
    
    @abstractmethod
    def delete(self, key: str) -> None:
        pass

# Implementations can mix and match
class ReadOnlyCache(Readable):
    """Only implements reading."""
    
    def read(self, key: str) -> bytes:
        return self.cache.get(key)

class FullCache(Readable, Writable, Deletable):
    """Implements all operations."""
    
    def read(self, key: str) -> bytes:
        return self.cache.get(key)
    
    def write(self, key: str, data: bytes) -> None:
        self.cache.set(key, data)
    
    def delete(self, key: str) -> None:
        self.cache.delete(key)
```

## 5. Dependency Inversion Principle (DIP)

### Definition
High-level modules should not depend on low-level modules. Both should depend on abstractions.

### Implementation Examples

#### ✅ Good: Dependency Injection
```python
# src/brain_go_brrr/services/pipeline.py
from abc import ABC, abstractmethod

# Abstractions
class ModelRepository(ABC):
    """Abstract model storage."""
    
    @abstractmethod
    def get_model(self, name: str) -> BaseEEGModel:
        pass

class DataRepository(ABC):
    """Abstract data storage."""
    
    @abstractmethod
    def save_results(self, job_id: str, results: Dict) -> None:
        pass

class NotificationService(ABC):
    """Abstract notification service."""
    
    @abstractmethod
    async def notify(self, message: str) -> None:
        pass

# High-level module depends on abstractions
class AnalysisPipeline:
    """High-level analysis orchestrator."""
    
    def __init__(
        self,
        model_repo: ModelRepository,
        data_repo: DataRepository,
        notifier: NotificationService
    ):
        # Depends on abstractions, not concrete implementations
        self.model_repo = model_repo
        self.data_repo = data_repo
        self.notifier = notifier
    
    async def analyze(self, job_id: str, eeg_path: Path) -> None:
        """Run analysis pipeline."""
        # Load model through abstraction
        model = self.model_repo.get_model("eegpt")
        
        # Process data
        results = await self._process_eeg(model, eeg_path)
        
        # Save through abstraction
        self.data_repo.save_results(job_id, results)
        
        # Notify through abstraction
        await self.notifier.notify(f"Job {job_id} completed")

# Concrete implementations
class S3ModelRepository(ModelRepository):
    """S3-based model storage."""
    
    def get_model(self, name: str) -> BaseEEGModel:
        # S3-specific implementation
        path = self._download_from_s3(name)
        return load_model(path)

class RedisDataRepository(DataRepository):
    """Redis-based result storage."""
    
    def save_results(self, job_id: str, results: Dict) -> None:
        # Redis-specific implementation
        self.redis.set(f"results:{job_id}", json.dumps(results))

class EmailNotificationService(NotificationService):
    """Email notification implementation."""
    
    async def notify(self, message: str) -> None:
        # Email-specific implementation
        await self.smtp.send_email(message)

# Dependency injection in practice
def create_pipeline() -> AnalysisPipeline:
    """Factory function with dependency injection."""
    return AnalysisPipeline(
        model_repo=S3ModelRepository(),
        data_repo=RedisDataRepository(),
        notifier=EmailNotificationService()
    )
```

#### ❌ Bad: Direct Dependencies
```python
class AnalysisPipeline:
    """Violates DIP - directly depends on concrete implementations."""
    
    def __init__(self):
        # Direct coupling to concrete classes
        self.s3_client = boto3.client('s3')
        self.redis_client = redis.Redis()
        self.smtp_server = smtplib.SMTP()
    
    def analyze(self, job_id: str, eeg_path: Path) -> None:
        # Tightly coupled to S3
        model_bytes = self.s3_client.get_object(
            Bucket='models',
            Key='eegpt.ckpt'
        )
        
        # Tightly coupled to Redis
        self.redis_client.set(f"results:{job_id}", results)
        
        # Tightly coupled to SMTP
        self.smtp_server.send_message(email)
```

### Repository Pattern with DIP
```python
# src/brain_go_brrr/repositories/interfaces.py
class EEGRepository(ABC):
    """Abstract repository for EEG data."""
    
    @abstractmethod
    async def get(self, file_id: str) -> mne.io.Raw:
        pass
    
    @abstractmethod
    async def save(self, file_id: str, raw: mne.io.Raw) -> None:
        pass

# src/brain_go_brrr/repositories/s3.py
class S3EEGRepository(EEGRepository):
    """S3 implementation of EEG repository."""
    
    def __init__(self, s3_client, bucket: str):
        self.s3 = s3_client
        self.bucket = bucket
    
    async def get(self, file_id: str) -> mne.io.Raw:
        # Download from S3 and load
        with tempfile.NamedTemporaryFile() as tmp:
            await self.s3.download_file(self.bucket, file_id, tmp.name)
            return mne.io.read_raw_edf(tmp.name)
    
    async def save(self, file_id: str, raw: mne.io.Raw) -> None:
        # Save to S3
        with tempfile.NamedTemporaryFile() as tmp:
            raw.save(tmp.name)
            await self.s3.upload_file(tmp.name, self.bucket, file_id)

# src/brain_go_brrr/services/analysis.py
class AnalysisService:
    """Service depends on abstraction."""
    
    def __init__(self, eeg_repo: EEGRepository):
        self.eeg_repo = eeg_repo  # Depends on interface
    
    async def analyze_file(self, file_id: str) -> AnalysisResult:
        # Works with any repository implementation
        raw = await self.eeg_repo.get(file_id)
        results = self._analyze(raw)
        return results
```

## Practical SOLID Implementation Guidelines

### 1. Service Layer Architecture
```python
# src/brain_go_brrr/services/orchestrator.py
class AnalysisOrchestrator:
    """Main orchestrator following all SOLID principles."""
    
    def __init__(
        self,
        quality_service: QualityAnalyzer,
        abnormal_detector: AbnormalityDetector,
        sleep_analyzer: Optional[SleepAnalyzer] = None,
        event_detector: Optional[EventDetector] = None,
        repository: DataRepository,
        notifier: NotificationService
    ):
        # SRP: Only orchestrates, doesn't implement analysis
        # OCP: Can add new analyzers without modification
        # LSP: All analyzers are interchangeable
        # ISP: Only depends on needed interfaces
        # DIP: Depends on abstractions
        self.quality_service = quality_service
        self.abnormal_detector = abnormal_detector
        self.sleep_analyzer = sleep_analyzer
        self.event_detector = event_detector
        self.repository = repository
        self.notifier = notifier
    
    async def process_eeg(self, job_id: str, config: AnalysisConfig) -> None:
        """Orchestrate analysis based on configuration."""
        try:
            # Load data
            raw = await self.repository.load_eeg(job_id)
            
            # Quality control (always run)
            quality_report = await self.quality_service.analyze_quality(raw)
            
            if quality_report.is_acceptable:
                # Abnormality detection (always run)
                abnormal_result = await self.abnormal_detector.detect_abnormality(
                    quality_report.cleaned_data
                )
                
                # Conditional analyses based on config
                if config.run_sleep_analysis and self.sleep_analyzer:
                    sleep_report = await self.sleep_analyzer.analyze_sleep(raw)
                
                if config.run_event_detection and abnormal_result.is_abnormal:
                    events = await self.event_detector.detect_events(raw)
            
            # Save results
            await self.repository.save_results(job_id, results)
            
            # Notify completion
            await self.notifier.notify(f"Analysis {job_id} completed")
            
        except Exception as e:
            await self.notifier.notify(f"Analysis {job_id} failed: {str(e)}")
            raise
```

### 2. Factory Pattern for Dependency Management
```python
# src/brain_go_brrr/factories.py
class ServiceFactory:
    """Factory for creating services with proper dependencies."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def create_analysis_pipeline(self) -> AnalysisOrchestrator:
        """Create fully configured analysis pipeline."""
        # Create repositories
        model_repo = self._create_model_repository()
        data_repo = self._create_data_repository()
        
        # Create services
        quality_service = self._create_quality_service()
        abnormal_detector = self._create_abnormal_detector(model_repo)
        sleep_analyzer = self._create_sleep_analyzer() if self.config.enable_sleep else None
        event_detector = self._create_event_detector() if self.config.enable_events else None
        notifier = self._create_notifier()
        
        return AnalysisOrchestrator(
            quality_service=quality_service,
            abnormal_detector=abnormal_detector,
            sleep_analyzer=sleep_analyzer,
            event_detector=event_detector,
            repository=data_repo,
            notifier=notifier
        )
    
    def _create_model_repository(self) -> ModelRepository:
        """Create appropriate model repository based on config."""
        if self.config.storage_backend == "s3":
            return S3ModelRepository(self.config.s3_config)
        elif self.config.storage_backend == "local":
            return LocalModelRepository(self.config.model_path)
        else:
            raise ValueError(f"Unknown storage backend: {self.config.storage_backend}")
```

### 3. Testing with SOLID
```python
# tests/test_orchestrator.py
class TestAnalysisOrchestrator:
    """Test orchestrator with mock dependencies."""
    
    def test_successful_analysis(self):
        # Create mocks following interfaces
        quality_service = Mock(spec=QualityAnalyzer)
        abnormal_detector = Mock(spec=AbnormalityDetector)
        repository = Mock(spec=DataRepository)
        notifier = Mock(spec=NotificationService)
        
        # Configure mocks
        quality_service.analyze_quality.return_value = QualityReport(is_acceptable=True)
        abnormal_detector.detect_abnormality.return_value = AbnormalityResult(score=0.8)
        
        # Create orchestrator with mocks
        orchestrator = AnalysisOrchestrator(
            quality_service=quality_service,
            abnormal_detector=abnormal_detector,
            repository=repository,
            notifier=notifier
        )
        
        # Test
        asyncio.run(orchestrator.process_eeg("test-job", AnalysisConfig()))
        
        # Verify interactions
        quality_service.analyze_quality.assert_called_once()
        abnormal_detector.detect_abnormality.assert_called_once()
        repository.save_results.assert_called_once()
        notifier.notify.assert_called_with("Analysis test-job completed")
```

## Benefits of SOLID in Brain-Go-Brrr

1. **Maintainability**: Each component has a clear, single purpose
2. **Testability**: Easy to mock dependencies and test in isolation
3. **Extensibility**: New features can be added without modifying existing code
4. **Flexibility**: Components can be swapped out easily
5. **Reusability**: Well-defined interfaces enable code reuse
6. **Team Collaboration**: Clear boundaries between components

## Common SOLID Patterns in the Codebase

### 1. Strategy Pattern (OCP)
- Event detection strategies
- Model selection strategies
- Preprocessing strategies

### 2. Repository Pattern (DIP)
- Data access abstraction
- Model storage abstraction
- Results caching abstraction

### 3. Factory Pattern (DIP)
- Service creation
- Model instantiation
- Pipeline configuration

### 4. Observer Pattern (OCP, DIP)
- Progress notifications
- Event subscriptions
- Result callbacks

### 5. Decorator Pattern (OCP, SRP)
- Logging decorators
- Caching decorators
- Retry decorators

## Conclusion

By following SOLID principles throughout Brain-Go-Brrr, we create a robust, maintainable, and extensible system that can evolve with changing requirements while maintaining stability and reliability in critical medical applications.