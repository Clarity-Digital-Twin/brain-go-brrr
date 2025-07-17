# Technology Stack Document
## EEGPT Clinical Decision Support System

### Document Control
- **Version**: 1.0.0
- **Status**: Active
- **Last Updated**: 2025-01-17
- **Purpose**: Central reference for all technology choices and versions

---

## Core Technologies

### Programming Languages
| Language | Version | Usage |
|----------|---------|-------|
| Python | 3.11+ | Backend, ML, API |
| TypeScript | 5.0+ | Frontend, Type Safety |
| SQL | PostgreSQL 15 | Database |
| YAML | 1.2 | Configuration |
| Gherkin | 6.0 | BDD Specifications |

### Frontend Stack
| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2+ | UI Framework |
| Next.js | 14.0+ | Full-stack React Framework |
| Material-UI | 5.14+ | Component Library |
| Redux Toolkit | 2.0+ | State Management |
| RTK Query | 2.0+ | Data Fetching |
| D3.js | 7.8+ | EEG Waveform Visualization |
| Chart.js | 4.4+ | Metrics Visualization |
| React Hook Form | 7.48+ | Form Management |
| Zod | 3.22+ | Schema Validation |

### Backend Stack
| Technology | Version | Purpose |
|------------|---------|---------|
| FastAPI | 0.109+ | REST API Framework |
| Pydantic | 2.5+ | Data Validation |
| SQLAlchemy | 2.0+ | ORM |
| Alembic | 1.13+ | Database Migrations |
| Celery | 5.3+ | Task Queue |
| Redis | 7.2+ | Cache & Queue Backend |
| PostgreSQL | 15+ | Primary Database |
| TimescaleDB | 2.13+ | Time-series Extension |

### Machine Learning Stack
| Technology | Version | Purpose |
|------------|---------|---------|
| PyTorch | 2.2+ | Deep Learning Framework |
| MNE-Python | 1.6+ | EEG Processing |
| NumPy | 1.26+ | Numerical Computing |
| SciPy | 1.12+ | Scientific Computing |
| scikit-learn | 1.4+ | Classical ML |
| pandas | 2.2+ | Data Manipulation |
| YASA | 0.6+ | Sleep Staging |
| einops | 0.7+ | Tensor Operations |
| transformers | 4.37+ | Hugging Face Models |

### DevOps & Infrastructure
| Technology | Version | Purpose |
|------------|---------|---------|
| Docker | 24+ | Containerization |
| Kubernetes | 1.28+ | Container Orchestration |
| Helm | 3.13+ | K8s Package Manager |
| Terraform | 1.6+ | Infrastructure as Code |
| GitHub Actions | N/A | CI/CD |
| ArgoCD | 2.9+ | GitOps Deployment |
| Nginx | 1.24+ | Reverse Proxy |
| Traefik | 3.0+ | Load Balancer |

### Monitoring & Observability
| Technology | Version | Purpose |
|------------|---------|---------|
| Prometheus | 2.48+ | Metrics Collection |
| Grafana | 10.2+ | Metrics Visualization |
| Loki | 2.9+ | Log Aggregation |
| Tempo | 2.3+ | Distributed Tracing |
| OpenTelemetry | 1.21+ | Observability Framework |
| Sentry | 23.12+ | Error Tracking |
| PagerDuty | API v2 | Incident Management |

### Security & Compliance
| Technology | Version | Purpose |
|------------|---------|---------|
| HashiCorp Vault | 1.15+ | Secrets Management |
| OAuth2/OIDC | 2.1 | Authentication |
| JWT | RFC 7519 | Token Format |
| mTLS | 1.3 | Service Communication |
| OWASP ZAP | 2.14+ | Security Testing |
| SonarQube | 10.3+ | Code Quality |
| Trivy | 0.48+ | Vulnerability Scanning |

### Development Tools
| Tool | Version | Purpose |
|------------|---------|---------|
| uv | 0.1+ | Python Package Manager |
| Poetry | 1.7+ | Dependency Management |
| Black | 23.12+ | Code Formatter |
| Ruff | 0.1+ | Linter |
| mypy | 1.8+ | Type Checker |
| pytest | 8.0+ | Testing Framework |
| pytest-asyncio | 0.23+ | Async Testing |
| pytest-bdd | 6.1+ | BDD Testing |
| pre-commit | 3.6+ | Git Hooks |

### Cloud Services (AWS Primary)
| Service | Purpose | Alternative (Azure) |
|---------|---------|-------------------|
| EKS | Kubernetes Hosting | AKS |
| RDS | PostgreSQL Database | Azure Database |
| S3 | Object Storage | Blob Storage |
| CloudFront | CDN | Azure CDN |
| Route 53 | DNS | Azure DNS |
| KMS | Encryption Keys | Key Vault |
| CloudWatch | Monitoring | Azure Monitor |
| Lambda | Serverless Functions | Functions |
| SQS | Message Queue | Service Bus |
| Cognito | User Management | AD B2C |

---

## Version Management Policy

### Dependency Updates
- **Security patches**: Applied immediately
- **Minor updates**: Monthly review and update
- **Major updates**: Quarterly planning with testing
- **Breaking changes**: Requires architecture review

### Python Package Versions
```toml
# pyproject.toml excerpt
[project]
requires-python = ">=3.11"

[tool.uv]
python = "3.11"

[dependencies]
# Exact versions for medical device compliance
torch = "==2.2.0"
mne = "==1.6.0"
fastapi = "==0.109.0"
# ... see full pyproject.toml
```

### Container Base Images
```dockerfile
# Production base images
FROM python:3.11-slim-bookworm AS python-base
FROM node:20-alpine AS node-base
FROM nginx:1.24-alpine AS nginx-base
```

---

## Integration Standards

### API Versioning
- URL path versioning: `/api/v1/`, `/api/v2/`
- Semantic versioning for releases
- Deprecation notices: 6 months minimum
- Backward compatibility for 2 major versions

### Database Compatibility
- PostgreSQL 15+ required features:
  - JSONB with path operators
  - Parallel query execution
  - Logical replication
- TimescaleDB for time-series data
- Read replicas for scaling

### Browser Support
- Chrome 120+ (primary)
- Firefox 115+ (ESR)
- Safari 16+
- Edge 120+
- Mobile: iOS Safari 16+, Chrome Android

---

## Model Requirements

### EEGPT Model Specifications
- **Framework**: PyTorch 2.0+
- **Architecture**: Vision Transformer
- **Size**: 10M parameters
- **Input**: 256 Hz, 4-second windows
- **GPU Memory**: 8GB minimum, 16GB recommended
- **Inference**: ONNX Runtime compatible

### Hardware Requirements
| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|------------|
| CPU | 8 cores | 16 cores | 32 cores |
| RAM | 32 GB | 64 GB | 128 GB |
| GPU | RTX 3060 (12GB) | RTX 4070 (16GB) | A100 (40GB) |
| Storage | 500 GB SSD | 1 TB NVMe | 2 TB NVMe RAID |

---

## Development Environment

### Local Development Setup
```bash
# Required tools
- Python 3.11+
- Node.js 20+
- Docker Desktop
- PostgreSQL 15
- Redis 7

# IDE Recommendations
- VS Code with Python/TS extensions
- PyCharm Professional
- DataGrip for database

# Required VS Code Extensions
- Python
- Pylance
- Black Formatter
- ESLint
- Prettier
- Docker
- GitLens
```

### Environment Variables
```bash
# .env.example
DATABASE_URL=postgresql://user:pass@localhost/eegpt
REDIS_URL=redis://localhost:6379
S3_BUCKET=eegpt-dev
JWT_SECRET_KEY=<generate-with-openssl>
EEGPT_MODEL_PATH=/data/models/eegpt/pretrained
LOG_LEVEL=DEBUG
CUDA_VISIBLE_DEVICES=0
```

---

## Compliance Considerations

### Medical Device Software
- IEC 62304 compliance for medical device software
- ISO 13485 quality management
- FDA 21 CFR Part 820 quality system regulation
- EU MDR 2017/745 for European deployment

### Data Protection
- HIPAA compliance (US healthcare)
- GDPR compliance (EU data protection)
- State-specific regulations (e.g., CCPA)
- Encryption standards: FIPS 140-2 Level 2

### Audit Requirements
- Immutable audit logs
- User action tracking
- Data access logging
- Change management records
- 7-year retention policy

---

## Migration Guidelines

### Database Migrations
```bash
# Using Alembic
alembic revision --autogenerate -m "description"
alembic upgrade head
alembic downgrade -1

# Migration testing required:
- Forward migration
- Rollback testing
- Data integrity checks
- Performance impact
```

### API Migrations
```python
# Versioned endpoints
@router.post("/api/v1/analyze")  # Current
@router.post("/api/v2/analyze")  # New version

# Deprecation headers
response.headers["Sunset"] = "Sat, 31 Dec 2025 23:59:59 GMT"
response.headers["Deprecation"] = "true"
```

---

## Cost Optimization

### Resource Allocation
- Auto-scaling for API servers (2-10 instances)
- GPU instances only for ML workers
- Spot instances for batch processing
- Reserved instances for baseline load
- S3 lifecycle policies for data archival

### Monitoring Costs
- CloudWatch: $0.30/metric/month
- Application metrics: ~$500/month
- Log storage: $0.50/GB
- Data transfer: Minimize cross-AZ traffic

---

## Disaster Recovery

### Backup Strategy
- Database: Daily snapshots, 30-day retention
- S3 data: Cross-region replication
- Model weights: Version controlled in Git LFS
- Configuration: GitOps with ArgoCD

### Recovery Targets
- RTO (Recovery Time Objective): 4 hours
- RPO (Recovery Point Objective): 1 hour
- Automated failover for critical services
- Manual failover for non-critical services

---

## Future Considerations

### Planned Upgrades
- PyTorch 2.3 (Q2 2025) - Compile improvements
- PostgreSQL 16 (Q3 2025) - Performance gains
- Kubernetes 1.30 (Q4 2025) - Security features
- React 19 (When stable) - Concurrent features

### Technology Evaluation
- Rust for performance-critical components
- WebAssembly for client-side processing
- GraphQL for flexible API queries
- gRPC for internal service communication

This document should be reviewed quarterly and updated as technology choices evolve.