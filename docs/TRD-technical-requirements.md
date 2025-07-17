# Technical Requirements Document (TRD)
## EEGPT Clinical Decision Support System

### Document Control
- **Version**: 1.0.0
- **Status**: Draft
- **Last Updated**: 2025-01-17
- **Owner**: Engineering Lead
- **References**: PRD-product-requirements.md, literature-master-reference.md

### System Architecture

#### High-Level Architecture
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Frontend  │────▶│    API Gateway   │────▶│  Load Balancer  │
│   (React/Next)  │     │    (Kong/NGINX)  │     │    (AWS ALB)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                        ┌──────────────────────────────────┴───────────┐
                        │                                              │
                  ┌─────▼────────┐                          ┌─────────▼────────┐
                  │  API Service  │                          │  Worker Service  │
                  │   (FastAPI)   │                          │    (Celery)      │
                  └───────┬───────┘                          └─────────┬────────┘
                          │                                            │
                  ┌───────▼───────┐                          ┌─────────▼────────┐
                  │   PostgreSQL  │                          │   Redis Queue    │
                  │  (Metadata)   │                          │  (Job Manager)   │
                  └───────────────┘                          └──────────────────┘
                                                                       │
                  ┌───────────────┐                          ┌─────────▼────────┐
                  │  S3 Storage   │◀─────────────────────────│   ML Pipeline    │
                  │  (EDF Files)  │                          │  (GPU Workers)   │
                  └───────────────┘                          └──────────────────┘
```

### Technical Stack

#### Frontend
- **Framework**: React 18+ with Next.js 14
- **UI Library**: Material-UI v5 / Ant Design
- **State Management**: Redux Toolkit + RTK Query
- **Visualization**: D3.js for EEG waveforms, Chart.js for metrics
- **Testing**: Jest, React Testing Library, Cypress

#### Backend
- **API Framework**: FastAPI 0.100+
- **Task Queue**: Celery 5.3+ with Redis
- **Database**: PostgreSQL 15+ with TimescaleDB
- **ORM**: SQLAlchemy 2.0+
- **Authentication**: OAuth2 with JWT tokens
- **API Documentation**: OpenAPI 3.0 auto-generated

#### ML/AI Stack
- **Runtime**: Python 3.11+
- **Deep Learning**: PyTorch 2.0+
- **EEG Processing**: MNE-Python 1.6+
- **Models**:
  - EEGPT (10M parameters)
  - YASA (sleep staging)
  - Autoreject (artifact removal)
- **Serving**: TorchServe / Triton Inference Server
- **Monitoring**: Weights & Biases, MLflow

#### Infrastructure
- **Container**: Docker 24+, Docker Compose
- **Orchestration**: Kubernetes 1.28+ (production)
- **Cloud**: AWS (primary), Azure (secondary)
- **CI/CD**: GitHub Actions, ArgoCD
- **Monitoring**: Prometheus, Grafana, ELK Stack
- **Security**: Vault for secrets, AWS KMS

### API Specifications

#### REST API Endpoints

```yaml
openapi: 3.0.0
info:
  title: EEGPT Clinical API
  version: 1.0.0

paths:
  /api/v1/auth/login:
    post:
      summary: Authenticate user
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                username: string
                password: string
      responses:
        200:
          description: JWT token
          
  /api/v1/eeg/upload:
    post:
      summary: Upload EEG file for analysis
      security:
        - bearerAuth: []
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                file:
                  type: string
                  format: binary
                metadata:
                  type: object
      responses:
        202:
          description: Job accepted
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/JobResponse'
                
  /api/v1/eeg/analyze/{job_id}:
    get:
      summary: Get analysis results
      parameters:
        - name: job_id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Analysis complete
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/AnalysisResult'
```

### Data Models

#### Database Schema
```sql
-- Core Tables
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE eeg_studies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(255) NOT NULL,
    uploaded_by UUID REFERENCES users(id),
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    study_id UUID REFERENCES eeg_studies(id),
    analysis_type VARCHAR(50) NOT NULL,
    result JSONB NOT NULL,
    confidence FLOAT,
    processing_time FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit table for compliance
CREATE TABLE audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Processing Pipeline

#### EEG Analysis Pipeline
```python
class EEGAnalysisPipeline:
    """Main processing pipeline adhering to SOLID principles"""
    
    def __init__(self, 
                 qc_analyzer: IQualityAnalyzer,
                 abnormal_detector: IAbnormalDetector,
                 event_detector: IEventDetector,
                 sleep_analyzer: ISleepAnalyzer):
        self.qc = qc_analyzer
        self.abnormal = abnormal_detector
        self.events = event_detector
        self.sleep = sleep_analyzer
    
    async def process(self, eeg_file_path: Path) -> AnalysisResult:
        # 1. Load and validate
        raw = await self.load_eeg(eeg_file_path)
        
        # 2. Quality control
        qc_result = await self.qc.analyze(raw)
        if qc_result.unusable:
            return AnalysisResult(status="failed", reason="poor_quality")
        
        # 3. Parallel analysis
        results = await asyncio.gather(
            self.abnormal.detect(raw),
            self.events.detect(raw),
            self.sleep.analyze(raw) if raw.duration > 600 else None
        )
        
        return AnalysisResult(
            qc=qc_result,
            abnormal=results[0],
            events=results[1],
            sleep=results[2]
        )
```

### Security Requirements

#### Authentication & Authorization
- OAuth2 with JWT tokens (RS256)
- Multi-factor authentication for clinical users
- Role-based access control (RBAC)
- Session timeout: 30 minutes
- API rate limiting: 100 requests/minute

#### Data Security
- Encryption at rest: AES-256
- Encryption in transit: TLS 1.3
- PHI de-identification for research mode
- Data retention: 7 years (configurable)
- Automated data purging

#### Compliance
- HIPAA Security Rule compliance
- 21 CFR Part 11 (electronic records)
- GDPR compliance for EU deployment
- SOC2 Type II certification

### Performance Requirements

#### Latency Targets
- API response: <100ms (p95)
- File upload: <5 seconds for 100MB
- QC analysis: <30 seconds
- Full analysis: <2 minutes
- Report generation: <10 seconds

#### Throughput
- Concurrent analyses: 50
- Daily volume: 10,000 studies
- Peak load: 100 studies/hour
- Storage: 10TB initial, scalable

### Deployment Architecture

#### Container Specifications
```yaml
# docker-compose.yml
version: '3.8'
services:
  api:
    image: eegpt/api:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
          
  worker:
    image: eegpt/worker:latest
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Monitoring & Observability

#### Metrics
- Application metrics: Prometheus
- Infrastructure metrics: CloudWatch/Azure Monitor
- Custom metrics: Processing time, accuracy, queue depth
- SLI/SLO tracking

#### Logging
- Structured logging: JSON format
- Log aggregation: ELK Stack
- Log retention: 90 days
- Sensitive data masking

#### Tracing
- Distributed tracing: OpenTelemetry
- Request correlation IDs
- Performance profiling

### Testing Strategy

#### Test Coverage Requirements
- Unit tests: >80% coverage
- Integration tests: All API endpoints
- End-to-end tests: Critical user flows
- Performance tests: Load/stress testing
- Security tests: OWASP top 10

#### Test Automation
```python
# Example test structure
class TestEEGPipeline:
    @pytest.fixture
    def mock_eeg_data(self):
        return create_mock_eeg(duration=600, channels=19)
    
    async def test_abnormal_detection_accuracy(self, mock_eeg_data):
        pipeline = EEGAnalysisPipeline()
        result = await pipeline.process(mock_eeg_data)
        
        assert result.abnormal.confidence > 0.8
        assert result.processing_time < 120  # seconds
```

### AI Agent Integration Guidelines

#### Code Documentation Standards
```python
"""
Module: eeg_analysis_pipeline.py
Purpose: Core EEG analysis pipeline implementing EEGPT-based clinical decision support
Requirements: FR1, FR2, FR3 from PRD-product-requirements.md
Dependencies: 
  - EEGPT model weights: /data/models/eegpt/pretrained/eegpt_large.pt
  - Config: /config/pipeline_config.yaml
  
AI Agent Instructions:
- This module follows SOLID principles with dependency injection
- All analyzers implement their respective interfaces (IQualityAnalyzer, etc.)
- Use async/await for all I/O operations
- Log all processing steps with correlation IDs
- Handle failures gracefully with specific error types
"""
```

#### Directory Structure
```
src/
├── api/                 # FastAPI application
│   ├── __init__.py
│   ├── main.py         # Application entry point
│   ├── routers/        # API route handlers
│   ├── models/         # Pydantic models
│   └── dependencies/   # Dependency injection
├── core/               # Business logic
│   ├── analyzers/      # Analysis modules
│   ├── interfaces/     # Abstract interfaces
│   └── exceptions/     # Custom exceptions
├── infrastructure/     # External services
│   ├── database/       # Database access
│   ├── storage/        # File storage
│   └── cache/          # Redis cache
└── tests/              # Test suite
```

### Development Workflow

#### Branch Strategy
- main: Production-ready code
- develop: Integration branch
- feature/*: New features
- bugfix/*: Bug fixes
- release/*: Release candidates

#### Code Review Requirements
- 2 approvals required
- Automated tests must pass
- Security scan must pass
- Documentation updated
- AI agent guidelines followed

### Approval
| Role | Name | Signature | Date |
|------|------|-----------|------|
| Engineering Lead | | | |
| Security Lead | | | |
| DevOps Lead | | | |
| QA Lead | | | |