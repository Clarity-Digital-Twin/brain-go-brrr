# Deployment Architecture for Brain-Go-Brrr

## Executive Summary

This document defines the production deployment architecture for Brain-Go-Brrr, designed for high availability, scalability, and HIPAA compliance. We use a containerized microservices architecture with Kubernetes orchestration, ensuring zero-downtime deployments and automatic scaling based on demand.

## Architecture Overview

### High-Level Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   CloudFront    │────▶│ Load Balancer   │────▶│   API Gateway   │
│      (CDN)      │     │     (ALB)       │     │   (Kong/Nginx)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
                              ┌──────────────────────────┼──────────────────────────┐
                              │                          │                          │
                              ▼                          ▼                          ▼
                     ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
                     │   API Service   │      │ Analysis Worker │      │  Admin Service  │
                     │   (FastAPI)     │      │   (Celery)      │      │   (Django)      │
                     └─────────────────┘      └─────────────────┘      └─────────────────┘
                              │                          │                          │
                  ┌──────────┴──────────┬───────────────┴──────────┬──────────────┘
                  │                     │                           │
                  ▼                     ▼                           ▼
         ┌─────────────────┐  ┌─────────────────┐      ┌─────────────────┐
         │   PostgreSQL    │  │     Redis       │      │   S3 Storage    │
         │  (TimescaleDB)  │  │ (Cache/Queue)   │      │  (EEG Files)    │
         └─────────────────┘  └─────────────────┘      └─────────────────┘
```

## Container Architecture

### 1. Docker Images
```dockerfile
# docker/api/Dockerfile
FROM python:3.11-slim AS base

# Security: Run as non-root user
RUN useradd -m -u 1000 appuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache .

# Multi-stage build for smaller image
FROM python:3.11-slim

# Copy from build stage
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Security setup
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --chown=appuser:appuser . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "brain_go_brrr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# docker/worker/Dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04 AS base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ML dependencies
WORKDIR /app
COPY requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy model files
COPY models/ /models/

# Application setup
COPY --chown=1000:1000 . .

USER 1000
CMD ["celery", "-A", "brain_go_brrr.workers", "worker", "--loglevel=info"]
```

### 2. Docker Compose (Development)
```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/braindb
      - REDIS_URL=redis://redis:6379
      - S3_ENDPOINT=http://minio:9000
      - ENV=development
    depends_on:
      - db
      - redis
      - minio
    volumes:
      - ./src:/app/src  # Hot reload in development
    
  worker:
    build:
      context: .
      dockerfile: docker/worker/Dockerfile
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/braindb
      - REDIS_URL=redis://redis:6379
      - S3_ENDPOINT=http://minio:9000
      - MODEL_PATH=/models
    depends_on:
      - db
      - redis
      - minio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    
  db:
    image: timescale/timescaledb:2.11.0-pg15
    environment:
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=braindb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass password
    volumes:
      - redis_data:/data
    
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

volumes:
  postgres_data:
  redis_data:
  minio_data:
```

## Kubernetes Deployment

### 1. Namespace and RBAC
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: brain-go-brrr
  labels:
    name: brain-go-brrr
    environment: production

---
# k8s/rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: brain-go-brrr
  name: brain-go-brrr-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: brain-go-brrr-sa
  namespace: brain-go-brrr

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: brain-go-brrr-rb
  namespace: brain-go-brrr
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: brain-go-brrr-role
subjects:
- kind: ServiceAccount
  name: brain-go-brrr-sa
  namespace: brain-go-brrr
```

### 2. API Deployment
```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: brain-go-brrr
  labels:
    app: api
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: brain-go-brrr-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: api
        image: brain-go-brrr/api:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          protocol: TCP
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: access_key_id
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: aws-credentials
              key: secret_access_key
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: api-config
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - api
              topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: brain-go-brrr
spec:
  selector:
    app: api
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
```

### 3. Worker Deployment (GPU)
```yaml
# k8s/worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analysis-worker
  namespace: brain-go-brrr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: analysis-worker
  template:
    metadata:
      labels:
        app: analysis-worker
    spec:
      serviceAccountName: brain-go-brrr-sa
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - name: worker
        image: brain-go-brrr/worker:latest
        imagePullPolicy: Always
        env:
        - name: WORKER_TYPE
          value: "gpu"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            cpu: 2000m
            memory: 8Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 4000m
            memory: 16Gi
            nvidia.com/gpu: 1
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: shared-memory
          mountPath: /dev/shm
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: shared-memory
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi
      nodeSelector:
        node.kubernetes.io/gpu: "true"
```

### 4. Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: brain-go-brrr
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60
      selectPolicy: Max

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-hpa
  namespace: brain-go-brrr
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: analysis-worker
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: celery_queue_length
        selector:
          matchLabels:
            queue: analysis
      target:
        type: Value
        value: "50"  # Scale up if queue > 50
```

## Infrastructure as Code

### 1. Terraform Configuration
```hcl
# terraform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "brain-go-brrr-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
    dynamodb_table = "terraform-state-lock"
  }
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  
  name = "brain-go-brrr-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]
  
  enable_nat_gateway = true
  single_nat_gateway = false  # HA NAT gateways
  enable_dns_hostnames = true
  enable_dns_support = true
  
  enable_flow_log = true
  flow_log_destination_type = "cloud-watch-logs"
  
  tags = {
    Environment = "production"
    Application = "brain-go-brrr"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "19.0.0"
  
  cluster_name    = "brain-go-brrr-cluster"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Control plane logging
  cluster_enabled_log_types = [
    "api", "audit", "authenticator", "controllerManager", "scheduler"
  ]
  
  # OIDC Provider for IRSA
  enable_irsa = true
  
  # Node groups
  eks_managed_node_groups = {
    general = {
      name = "general-workers"
      
      instance_types = ["m5.xlarge"]
      
      min_size     = 3
      max_size     = 10
      desired_size = 5
      
      disk_size = 100
      disk_type = "gp3"
      
      labels = {
        Environment = "production"
        NodeType    = "general"
      }
      
      tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/brain-go-brrr-cluster" = "owned"
      }
    }
    
    gpu = {
      name = "gpu-workers"
      
      instance_types = ["g4dn.xlarge"]  # NVIDIA T4 GPU
      
      min_size     = 1
      max_size     = 5
      desired_size = 2
      
      disk_size = 200
      disk_type = "gp3"
      
      labels = {
        Environment = "production"
        NodeType    = "gpu"
        "node.kubernetes.io/gpu" = "true"
      }
      
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
      
      # GPU-specific user data
      user_data = base64encode(templatefile("${path.module}/gpu-init.sh", {}))
    }
  }
}

# RDS (PostgreSQL with TimescaleDB)
module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "6.0.0"
  
  identifier = "brain-go-brrr-db"
  
  engine            = "postgres"
  engine_version    = "15.3"
  instance_class    = "db.r6g.2xlarge"
  allocated_storage = 500
  storage_type      = "gp3"
  storage_encrypted = true
  
  # High availability
  multi_az = true
  
  # Database configuration
  db_name  = "braindb"
  username = "brainadmin"
  port     = "5432"
  
  # VPC configuration
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  
  # Backup configuration
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  # Monitoring
  enabled_cloudwatch_logs_exports = ["postgresql"]
  create_cloudwatch_log_group     = true
  
  # Parameter group for TimescaleDB
  parameter_group_name = aws_db_parameter_group.timescale.name
  
  tags = {
    Environment = "production"
    Application = "brain-go-brrr"
  }
}

# ElastiCache (Redis)
module "elasticache" {
  source = "terraform-aws-modules/elasticache/aws"
  version = "1.0.0"
  
  cluster_id = "brain-go-brrr-cache"
  
  engine          = "redis"
  engine_version  = "7.0"
  node_type       = "cache.r6g.xlarge"
  num_cache_nodes = 1
  
  # Redis cluster mode
  parameter_group_family = "redis7"
  
  # High availability
  automatic_failover_enabled = true
  multi_az_enabled          = true
  num_node_groups           = 3
  replicas_per_node_group   = 2
  
  # Security
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token_enabled        = true
  
  # VPC configuration
  subnet_ids = module.vpc.private_subnets
  security_group_ids = [aws_security_group.redis.id]
  
  # Backup
  snapshot_retention_limit = 7
  snapshot_window         = "03:00-05:00"
  
  tags = {
    Environment = "production"
    Application = "brain-go-brrr"
  }
}

# S3 Buckets
resource "aws_s3_bucket" "eeg_files" {
  bucket = "brain-go-brrr-eeg-files-prod"
  
  tags = {
    Environment = "production"
    DataType    = "PHI"
  }
}

resource "aws_s3_bucket_versioning" "eeg_files" {
  bucket = aws_s3_bucket.eeg_files.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "eeg_files" {
  bucket = aws_s3_bucket.eeg_files.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.s3_key.arn
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "eeg_files" {
  bucket = aws_s3_bucket.eeg_files.id
  
  rule {
    id     = "archive_old_files"
    status = "Enabled"
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
  }
}
```

### 2. Helm Charts
```yaml
# helm/brain-go-brrr/Chart.yaml
apiVersion: v2
name: brain-go-brrr
description: Brain-Go-Brrr EEG Analysis Platform
type: application
version: 1.0.0
appVersion: "1.0.0"

dependencies:
  - name: postgresql
    version: 12.0.0
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: 17.0.0
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
  - name: prometheus
    version: 19.0.0
    repository: https://prometheus-community.github.io/helm-charts
    condition: monitoring.prometheus.enabled
  - name: grafana
    version: 6.50.0
    repository: https://grafana.github.io/helm-charts
    condition: monitoring.grafana.enabled

# helm/brain-go-brrr/values.yaml
replicaCount:
  api: 3
  worker: 2

image:
  api:
    repository: brain-go-brrr/api
    tag: latest
    pullPolicy: Always
  worker:
    repository: brain-go-brrr/worker
    tag: latest
    pullPolicy: Always

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
  hosts:
    - host: api.brain-go-brrr.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: brain-go-brrr-tls
      hosts:
        - api.brain-go-brrr.com

resources:
  api:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi
  worker:
    requests:
      cpu: 2000m
      memory: 8Gi
      nvidia.com/gpu: 1
    limits:
      cpu: 4000m
      memory: 16Gi
      nvidia.com/gpu: 1

autoscaling:
  api:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80
  worker:
    enabled: true
    minReplicas: 1
    maxReplicas: 10
    metrics:
      - type: External
        external:
          metric:
            name: celery_queue_length
          target:
            type: Value
            value: "50"

postgresql:
  enabled: false  # Using external RDS
  external:
    host: brain-go-brrr-db.region.rds.amazonaws.com
    port: 5432
    database: braindb
    existingSecret: database-credentials

redis:
  enabled: false  # Using external ElastiCache
  external:
    host: brain-go-brrr-cache.region.cache.amazonaws.com
    port: 6379
    existingSecret: redis-credentials

monitoring:
  prometheus:
    enabled: true
    retention: 30d
  grafana:
    enabled: true
    adminPassword: changeme
    dashboards:
      - brain-go-brrr-overview
      - api-performance
      - worker-performance
      - gpu-utilization
```

## CI/CD Pipeline

### 1. GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  AWS_REGION: us-east-1
  ECR_REGISTRY: ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ env.AWS_REGION }}.amazonaws.com
  EKS_CLUSTER: brain-go-brrr-cluster

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install uv
        uv sync --dev
    
    - name: Run tests
      run: |
        uv run pytest --cov=brain_go_brrr
    
    - name: Security scan
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        severity: 'CRITICAL,HIGH'

  build:
    needs: test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [api, worker]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
    
    - name: Build and push image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $ECR_REGISTRY/brain-go-brrr/${{ matrix.service }}:$IMAGE_TAG \
          -f docker/${{ matrix.service }}/Dockerfile .
        
        docker tag $ECR_REGISTRY/brain-go-brrr/${{ matrix.service }}:$IMAGE_TAG \
          $ECR_REGISTRY/brain-go-brrr/${{ matrix.service }}:latest
        
        docker push $ECR_REGISTRY/brain-go-brrr/${{ matrix.service }}:$IMAGE_TAG
        docker push $ECR_REGISTRY/brain-go-brrr/${{ matrix.service }}:latest
    
    - name: Image security scan
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.ECR_REGISTRY }}/brain-go-brrr/${{ matrix.service }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --region ${{ env.AWS_REGION }} --name ${{ env.EKS_CLUSTER }}
    
    - name: Install Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'
    
    - name: Deploy with Helm
      run: |
        helm upgrade --install brain-go-brrr ./helm/brain-go-brrr \
          --namespace brain-go-brrr \
          --set image.api.tag=${{ github.sha }} \
          --set image.worker.tag=${{ github.sha }} \
          --values helm/brain-go-brrr/values.production.yaml \
          --wait --timeout 10m
    
    - name: Verify deployment
      run: |
        kubectl rollout status deployment/api -n brain-go-brrr
        kubectl rollout status deployment/analysis-worker -n brain-go-brrr
    
    - name: Run smoke tests
      run: |
        API_URL=$(kubectl get ingress -n brain-go-brrr -o jsonpath='{.items[0].spec.rules[0].host}')
        curl -f https://$API_URL/health || exit 1
```

### 2. ArgoCD Configuration
```yaml
# argocd/application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: brain-go-brrr
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/org/brain-go-brrr
    targetRevision: HEAD
    path: helm/brain-go-brrr
    helm:
      valueFiles:
      - values.production.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: brain-go-brrr
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
    - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

## Monitoring & Observability

### 1. Prometheus Configuration
```yaml
# monitoring/prometheus-values.yaml
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
    
    serviceMonitorSelector:
      matchLabels:
        app.kubernetes.io/part-of: brain-go-brrr
    
    additionalScrapeConfigs:
    - job_name: 'brain-go-brrr-api'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - brain-go-brrr
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: api
        action: keep
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### 2. Grafana Dashboards
```json
// monitoring/dashboards/api-performance.json
{
  "dashboard": {
    "title": "Brain-Go-Brrr API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "sum(rate(http_requests_total{app=\"api\"}[5m])) by (method, status)"
        }]
      },
      {
        "title": "Response Time (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{app=\"api\"}[5m])) by (le, endpoint))"
        }]
      },
      {
        "title": "Active Analyses",
        "targets": [{
          "expr": "sum(eeg_active_analyses{app=\"api\"})"
        }]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "sum(rate(http_requests_total{app=\"api\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{app=\"api\"}[5m]))"
        }]
      }
    ]
  }
}
```

### 3. Logging Stack
```yaml
# monitoring/fluent-bit-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: brain-go-brrr
data:
  fluent-bit.conf: |
    [SERVICE]
        Flush         5
        Log_Level     info
        Daemon        off
    
    [INPUT]
        Name              tail
        Tag               kube.*
        Path              /var/log/containers/*brain-go-brrr*.log
        Parser            docker
        DB                /var/log/flb-kube.db
        Mem_Buf_Limit     50MB
        Skip_Long_Lines   On
    
    [FILTER]
        Name                kubernetes
        Match               kube.*
        Kube_URL            https://kubernetes.default.svc:443
        Kube_CA_File        /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        Kube_Token_File     /var/run/secrets/kubernetes.io/serviceaccount/token
        Merge_Log           On
        Keep_Log            Off
    
    [FILTER]
        Name                record_modifier
        Match               *
        Record environment  production
        Record application  brain-go-brrr
    
    [OUTPUT]
        Name                cloudwatch_logs
        Match               *
        region              us-east-1
        log_group_name      /aws/eks/brain-go-brrr/containers
        log_stream_prefix   ${HOSTNAME}-
        auto_create_group   true
```

## Security Hardening

### 1. Network Policies
```yaml
# k8s/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: brain-go-brrr
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:  # Allow DNS
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: UDP
      port: 53

---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-default
  namespace: brain-go-brrr
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

### 2. Pod Security Policies
```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: brain-go-brrr-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
  - ALL
  volumes:
  - configMap
  - emptyDir
  - projected
  - secret
  - persistentVolumeClaim
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  supplementalGroups:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  readOnlyRootFilesystem: true
```

## Disaster Recovery

### 1. Backup Strategy
```yaml
# k8s/backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
  namespace: brain-go-brrr
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: database-credentials
                  key: password
            command:
            - /bin/sh
            - -c
            - |
              DATE=$(date +%Y%m%d_%H%M%S)
              pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME | \
                gzip | \
                aws s3 cp - s3://brain-go-brrr-backups/postgres/$DATE.sql.gz
              
              # Cleanup old backups (keep 30 days)
              aws s3 ls s3://brain-go-brrr-backups/postgres/ | \
                while read -r line; do
                  createDate=$(echo $line | awk '{print $1" "$2}')
                  createDate=$(date -d "$createDate" +%s)
                  olderThan=$(date -d "30 days ago" +%s)
                  if [[ $createDate -lt $olderThan ]]; then
                    fileName=$(echo $line | awk '{print $4}')
                    aws s3 rm s3://brain-go-brrr-backups/postgres/$fileName
                  fi
                done
          restartPolicy: OnFailure
```

### 2. Multi-Region Failover
```hcl
# terraform/disaster-recovery.tf
# Cross-region replication for S3
resource "aws_s3_bucket_replication_configuration" "eeg_files" {
  role   = aws_iam_role.replication.arn
  bucket = aws_s3_bucket.eeg_files.id
  
  rule {
    id     = "replicate-to-dr-region"
    status = "Enabled"
    
    destination {
      bucket        = aws_s3_bucket.eeg_files_dr.arn
      storage_class = "STANDARD_IA"
      
      encryption_configuration {
        replica_kms_key_id = aws_kms_key.s3_key_dr.arn
      }
    }
  }
}

# RDS Read Replica in DR region
resource "aws_db_instance" "dr_replica" {
  provider = aws.dr_region
  
  replicate_source_db = module.rds.db_instance_id
  
  instance_class = "db.r6g.xlarge"
  
  # Enable automated backups in DR region
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  
  tags = {
    Environment = "production-dr"
    Application = "brain-go-brrr"
  }
}
```

## Performance Optimization

### 1. CDN Configuration
```hcl
# terraform/cloudfront.tf
resource "aws_cloudfront_distribution" "api_cdn" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Brain-Go-Brrr API CDN"
  default_root_object = ""
  
  origin {
    domain_name = aws_lb.alb.dns_name
    origin_id   = "ALB-brain-go-brrr"
    
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2", "TLSv1.3"]
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods   = ["GET", "HEAD", "OPTIONS"]
    target_origin_id = "ALB-brain-go-brrr"
    
    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type", "X-API-Version"]
      
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 86400
  }
  
  # Cache static assets
  ordered_cache_behavior {
    path_pattern     = "/static/*"
    allowed_methods  = ["GET", "HEAD"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "ALB-brain-go-brrr"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    min_ttl                = 86400
    default_ttl            = 604800
    max_ttl                = 31536000
    viewer_protocol_policy = "redirect-to-https"
    compress               = true
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "whitelist"
      locations        = ["US", "CA", "GB", "DE", "FR", "JP", "AU"]
    }
  }
  
  viewer_certificate {
    acm_certificate_arn = aws_acm_certificate.cert.arn
    ssl_support_method  = "sni-only"
  }
}
```

### 2. Database Optimization
```sql
-- init.sql for TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertable for time-series metrics
CREATE TABLE analysis_metrics (
    time TIMESTAMPTZ NOT NULL,
    job_id UUID NOT NULL,
    metric_name TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('analysis_metrics', 'time');

-- Create indexes for common queries
CREATE INDEX ON analysis_metrics (job_id, time DESC);
CREATE INDEX ON analysis_metrics (metric_name, time DESC);

-- Compression policy
ALTER TABLE analysis_metrics SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'job_id'
);

SELECT add_compression_policy('analysis_metrics', INTERVAL '7 days');

-- Retention policy
SELECT add_retention_policy('analysis_metrics', INTERVAL '90 days');

-- Continuous aggregates for dashboards
CREATE MATERIALIZED VIEW analysis_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    metric_name,
    avg(value) as avg_value,
    max(value) as max_value,
    min(value) as min_value,
    count(*) as count
FROM analysis_metrics
GROUP BY hour, metric_name;

SELECT add_continuous_aggregate_policy('analysis_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

## Conclusion

This deployment architecture provides a robust, scalable, and secure platform for Brain-Go-Brrr in production. The use of Kubernetes, infrastructure as code, and comprehensive monitoring ensures high availability and easy maintenance while meeting HIPAA compliance requirements.