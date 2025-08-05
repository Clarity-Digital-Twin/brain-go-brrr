# API Design Patterns for Brain-Go-Brrr

## Executive Summary

This document defines RESTful API design patterns for the Brain-Go-Brrr EEG analysis platform. We follow industry best practices for medical data APIs, ensuring security, reliability, and developer-friendly interfaces while maintaining HIPAA compliance.

## Core API Design Principles

### 1. RESTful Architecture
- **Resource-Based**: URLs identify resources, not actions
- **Stateless**: Each request contains all necessary information
- **Uniform Interface**: Consistent patterns across all endpoints
- **HATEOAS**: Hypermedia as the engine of application state

### 2. Medical API Requirements
- **HIPAA Compliance**: PHI protection at all levels
- **Audit Trails**: Complete logging of all operations
- **Idempotency**: Safe retry mechanisms
- **Rate Limiting**: Prevent abuse and ensure fairness

## API Structure

### Base URL Structure
```
https://api.brain-go-brrr.com/v1/
├── /auth           # Authentication endpoints
├── /eeg            # EEG file operations
├── /analyses       # Analysis operations
├── /results        # Result retrieval
├── /patients       # Patient management (PHI)
├── /reports        # Report generation
└── /admin          # Administrative operations
```

## Authentication & Authorization

### JWT-Based Authentication
```python
# src/brain_go_brrr/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import secrets

router = APIRouter(prefix="/auth", tags=["authentication"])

class AuthConfig:
    SECRET_KEY = secrets.token_urlsafe(32)
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

@router.post("/token", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    """Authenticate user and return JWT tokens."""
    # Verify credentials
    user = await authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.id, "scopes": user.scopes}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.id}
    )
    
    # Log authentication event
    await log_auth_event(db, user.id, "login", request.client.host)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Validate JWT token and return current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, 
            AuthConfig.SECRET_KEY, 
            algorithms=[AuthConfig.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Check token expiration
        exp = payload.get("exp")
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = await get_user(db, user_id=user_id)
    if user is None:
        raise credentials_exception
        
    return user
```

### Role-Based Access Control (RBAC)
```python
# src/brain_go_brrr/api/permissions.py
from enum import Enum
from typing import List

class Role(str, Enum):
    ADMIN = "admin"
    CLINICIAN = "clinician"
    TECHNICIAN = "technician"
    RESEARCHER = "researcher"
    PATIENT = "patient"

class Permission(str, Enum):
    READ_EEG = "read:eeg"
    WRITE_EEG = "write:eeg"
    DELETE_EEG = "delete:eeg"
    
    READ_ANALYSIS = "read:analysis"
    WRITE_ANALYSIS = "write:analysis"
    
    READ_PHI = "read:phi"
    WRITE_PHI = "write:phi"
    
    ADMIN_USERS = "admin:users"
    ADMIN_SYSTEM = "admin:system"

# Role-permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],  # All permissions
    Role.CLINICIAN: [
        Permission.READ_EEG,
        Permission.WRITE_EEG,
        Permission.READ_ANALYSIS,
        Permission.WRITE_ANALYSIS,
        Permission.READ_PHI,
        Permission.WRITE_PHI
    ],
    Role.TECHNICIAN: [
        Permission.READ_EEG,
        Permission.WRITE_EEG,
        Permission.READ_ANALYSIS
    ],
    Role.RESEARCHER: [
        Permission.READ_EEG,
        Permission.READ_ANALYSIS
        # No PHI access
    ],
    Role.PATIENT: [
        Permission.READ_EEG,
        Permission.READ_ANALYSIS
        # Only their own data
    ]
}

def require_permission(permission: Permission):
    """Decorator to check permissions."""
    async def permission_checker(
        current_user: User = Depends(get_current_user)
    ):
        user_permissions = ROLE_PERMISSIONS.get(current_user.role, [])
        
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission}"
            )
        
        # Additional checks for patient role
        if current_user.role == Role.PATIENT:
            # Ensure accessing only their own data
            # This check happens in the endpoint
            pass
            
        return current_user
    
    return permission_checker
```

## Core API Endpoints

### 1. EEG File Management
```python
# src/brain_go_brrr/api/eeg.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
import aiofiles
import hashlib

router = APIRouter(prefix="/eeg", tags=["eeg"])

class EEGUploadResponse(BaseModel):
    file_id: str
    filename: str
    size_bytes: int
    checksum: str
    upload_timestamp: datetime
    status: str = "uploaded"

class EEGMetadata(BaseModel):
    duration_seconds: float
    sampling_rate: float
    num_channels: int
    channel_names: List[str]
    patient_id: Optional[str] = None
    recording_date: Optional[datetime] = None
    technician_notes: Optional[str] = None

@router.post(
    "/upload",
    response_model=EEGUploadResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_eeg(
    file: UploadFile = File(...),
    metadata: Optional[EEGMetadata] = None,
    current_user: User = Depends(require_permission(Permission.WRITE_EEG)),
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage)
):
    """Upload EEG file for analysis."""
    # Validate file type
    if not file.filename.lower().endswith(('.edf', '.bdf', '.fif')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported: .edf, .bdf, .fif"
        )
    
    # Check file size
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    sha256_hash = hashlib.sha256()
    
    # Stream file to storage and calculate checksum
    file_id = generate_file_id()
    storage_path = f"eeg/{current_user.organization_id}/{file_id}"
    
    async with aiofiles.tempfile.NamedTemporaryFile(delete=False) as tmp:
        while chunk := await file.read(chunk_size):
            await tmp.write(chunk)
            sha256_hash.update(chunk)
            file_size += len(chunk)
            
            # Check size limit (2GB)
            if file_size > 2 * 1024 * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_ENTITY_TOO_LARGE,
                    detail="File too large. Maximum size: 2GB"
                )
        
        # Validate EDF structure
        try:
            validator = EDFValidator()
            validation_result = validator.validate_edf(Path(tmp.name))
            
            if not validation_result["valid"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid EDF file: {validation_result['errors']}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"EDF validation failed: {str(e)}"
            )
        
        # Upload to storage
        await storage.upload_file(tmp.name, storage_path)
    
    # Save to database
    eeg_file = EEGFile(
        id=file_id,
        filename=file.filename,
        size_bytes=file_size,
        checksum=sha256_hash.hexdigest(),
        storage_path=storage_path,
        uploaded_by=current_user.id,
        organization_id=current_user.organization_id,
        metadata=metadata.dict() if metadata else None
    )
    
    db.add(eeg_file)
    await db.commit()
    
    # Log upload event
    await log_audit_event(
        db,
        user_id=current_user.id,
        action="eeg.upload",
        resource_id=file_id,
        details={"filename": file.filename, "size": file_size}
    )
    
    return EEGUploadResponse(
        file_id=file_id,
        filename=file.filename,
        size_bytes=file_size,
        checksum=eeg_file.checksum,
        upload_timestamp=eeg_file.created_at
    )

@router.get("/{file_id}", response_model=EEGFileDetails)
async def get_eeg_details(
    file_id: str,
    current_user: User = Depends(require_permission(Permission.READ_EEG)),
    db: AsyncSession = Depends(get_db)
):
    """Get EEG file details and metadata."""
    eeg_file = await get_eeg_file(db, file_id)
    
    if not eeg_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EEG file not found"
        )
    
    # Check access permissions
    if not await can_access_file(current_user, eeg_file):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this file"
        )
    
    return EEGFileDetails.from_orm(eeg_file)

@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_eeg(
    file_id: str,
    current_user: User = Depends(require_permission(Permission.DELETE_EEG)),
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage)
):
    """Delete EEG file (soft delete for audit trail)."""
    eeg_file = await get_eeg_file(db, file_id)
    
    if not eeg_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EEG file not found"
        )
    
    # Soft delete
    eeg_file.deleted_at = datetime.utcnow()
    eeg_file.deleted_by = current_user.id
    
    await db.commit()
    
    # Schedule storage cleanup after retention period
    await schedule_storage_cleanup(
        storage_path=eeg_file.storage_path,
        cleanup_after_days=90  # HIPAA retention
    )
    
    # Log deletion
    await log_audit_event(
        db,
        user_id=current_user.id,
        action="eeg.delete",
        resource_id=file_id,
        details={"filename": eeg_file.filename}
    )
```

### 2. Analysis Operations
```python
# src/brain_go_brrr/api/analyses.py
from fastapi import APIRouter, BackgroundTasks
from typing import List, Optional, Dict, Any
from enum import Enum

router = APIRouter(prefix="/analyses", tags=["analyses"])

class AnalysisType(str, Enum):
    QUICK = "quick"  # QC + abnormality only
    STANDARD = "standard"  # QC + abnormality + sleep
    COMPREHENSIVE = "comprehensive"  # All analyses
    CUSTOM = "custom"  # User-selected analyses

class AnalysisRequest(BaseModel):
    file_id: str
    analysis_type: AnalysisType = AnalysisType.STANDARD
    
    # Optional parameters
    priority: Optional[str] = "normal"  # normal, expedite, urgent
    
    # Custom analysis options
    enable_quality_control: bool = True
    enable_abnormality_detection: bool = True
    enable_sleep_analysis: Optional[bool] = None
    enable_event_detection: Optional[bool] = None
    enable_seizure_detection: Optional[bool] = None
    
    # Processing options
    use_gpu: bool = True
    batch_size: Optional[int] = None
    
    # Clinical context (helps with interpretation)
    clinical_notes: Optional[str] = None
    suspected_conditions: Optional[List[str]] = None
    medications: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    job_id: str
    status: str = "queued"
    estimated_completion_time: Optional[datetime] = None
    queue_position: Optional[int] = None
    links: Dict[str, str]  # HATEOAS links

@router.post(
    "/",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permission(Permission.WRITE_ANALYSIS)),
    db: AsyncSession = Depends(get_db),
    queue: QueueService = Depends(get_queue)
):
    """Submit EEG file for analysis."""
    # Verify file access
    eeg_file = await get_eeg_file(db, request.file_id)
    if not eeg_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="EEG file not found"
        )
    
    if not await can_access_file(current_user, eeg_file):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this file"
        )
    
    # Create analysis job
    job_id = generate_job_id()
    
    analysis_job = AnalysisJob(
        id=job_id,
        file_id=request.file_id,
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        analysis_type=request.analysis_type,
        priority=request.priority,
        configuration=request.dict(exclude={'file_id', 'analysis_type', 'priority'}),
        status="queued"
    )
    
    db.add(analysis_job)
    await db.commit()
    
    # Submit to processing queue
    queue_position = await queue.submit_job(
        job_id=job_id,
        job_type="analysis",
        priority=request.priority,
        payload={
            "file_id": request.file_id,
            "analysis_config": analysis_job.configuration
        }
    )
    
    # Estimate completion time
    estimated_time = await estimate_completion_time(
        queue_position=queue_position,
        analysis_type=request.analysis_type,
        file_size=eeg_file.size_bytes
    )
    
    # Background task for notifications
    if current_user.notification_preferences.get("analysis_complete"):
        background_tasks.add_task(
            schedule_completion_notification,
            job_id=job_id,
            user_id=current_user.id,
            estimated_time=estimated_time
        )
    
    return AnalysisResponse(
        job_id=job_id,
        status="queued",
        estimated_completion_time=estimated_time,
        queue_position=queue_position,
        links={
            "self": f"/analyses/{job_id}",
            "status": f"/analyses/{job_id}/status",
            "cancel": f"/analyses/{job_id}/cancel",
            "results": f"/results/{job_id}"
        }
    )

@router.get("/{job_id}/status", response_model=JobStatus)
async def get_analysis_status(
    job_id: str,
    current_user: User = Depends(require_permission(Permission.READ_ANALYSIS)),
    db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """Get current status of analysis job."""
    # Try cache first
    cached_status = await cache.get(f"job_status:{job_id}")
    if cached_status:
        return JobStatus(**cached_status)
    
    # Get from database
    job = await get_analysis_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
    
    if not await can_access_job(current_user, job):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this job"
        )
    
    # Get detailed status
    status = JobStatus(
        job_id=job_id,
        status=job.status,
        progress_percent=job.progress_percent,
        current_stage=job.current_stage,
        stages_completed=job.stages_completed,
        stages_total=job.stages_total,
        started_at=job.started_at,
        estimated_completion=job.estimated_completion,
        messages=job.status_messages or []
    )
    
    # Cache for 5 seconds
    await cache.set(f"job_status:{job_id}", status.dict(), ttl=5)
    
    return status

@router.post("/{job_id}/cancel", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_analysis(
    job_id: str,
    current_user: User = Depends(require_permission(Permission.WRITE_ANALYSIS)),
    db: AsyncSession = Depends(get_db),
    queue: QueueService = Depends(get_queue)
):
    """Cancel running or queued analysis."""
    job = await get_analysis_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
    
    if job.status in ["completed", "failed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in {job.status} state"
        )
    
    # Cancel in queue
    cancelled = await queue.cancel_job(job_id)
    
    if cancelled:
        job.status = "cancelled"
        job.cancelled_at = datetime.utcnow()
        job.cancelled_by = current_user.id
        await db.commit()
        
        # Log cancellation
        await log_audit_event(
            db,
            user_id=current_user.id,
            action="analysis.cancel",
            resource_id=job_id
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job is currently processing and cannot be cancelled"
        )
```

### 3. Results Retrieval
```python
# src/brain_go_brrr/api/results.py
from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse
import json

router = APIRouter(prefix="/results", tags=["results"])

class AnalysisResults(BaseModel):
    job_id: str
    file_id: str
    completed_at: datetime
    analysis_type: str
    
    # Quality control results
    quality_report: Optional[QualityReport] = None
    
    # Abnormality detection results
    abnormality_detection: Optional[AbnormalityResult] = None
    
    # Sleep analysis results
    sleep_analysis: Optional[SleepAnalysisResult] = None
    
    # Event detection results
    event_detection: Optional[EventDetectionResult] = None
    
    # Processing metadata
    processing_time_seconds: float
    model_versions: Dict[str, str]
    
    # Clinical summary
    clinical_summary: Optional[str] = None
    recommendations: Optional[List[str]] = None

@router.get("/{job_id}", response_model=AnalysisResults)
async def get_results(
    job_id: str,
    format: Optional[str] = "json",  # json, pdf, dicom
    include_raw_features: bool = False,
    current_user: User = Depends(require_permission(Permission.READ_ANALYSIS)),
    db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache)
):
    """Retrieve analysis results."""
    # Check cache first
    cache_key = f"results:{job_id}:{format}:{include_raw_features}"
    cached = await cache.get(cache_key)
    
    if cached and format == "json":
        return AnalysisResults(**cached)
    
    # Get job and results
    job = await get_analysis_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found"
        )
    
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job status is {job.status}, not completed"
        )
    
    if not await can_access_job(current_user, job):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to these results"
        )
    
    # Load results from storage
    results = await load_results_from_storage(job.results_path)
    
    # Filter based on user preferences
    if not include_raw_features:
        # Remove large raw feature arrays
        results = filter_raw_features(results)
    
    # Format conversion
    if format == "pdf":
        pdf_bytes = await generate_pdf_report(results)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=analysis_{job_id}.pdf"
            }
        )
    elif format == "dicom":
        dicom_bytes = await generate_dicom_sr(results)
        return StreamingResponse(
            io.BytesIO(dicom_bytes),
            media_type="application/dicom",
            headers={
                "Content-Disposition": f"attachment; filename=analysis_{job_id}.dcm"
            }
        )
    
    # Cache JSON results
    await cache.set(cache_key, results.dict(), ttl=3600)  # 1 hour
    
    # Log access
    await log_audit_event(
        db,
        user_id=current_user.id,
        action="results.access",
        resource_id=job_id,
        details={"format": format}
    )
    
    return results

@router.get("/{job_id}/summary", response_model=ResultsSummary)
async def get_results_summary(
    job_id: str,
    current_user: User = Depends(require_permission(Permission.READ_ANALYSIS)),
    db: AsyncSession = Depends(get_db)
):
    """Get a concise summary of results."""
    results = await get_results(job_id, current_user=current_user, db=db)
    
    summary = ResultsSummary(
        job_id=job_id,
        is_abnormal=results.abnormality_detection.is_abnormal,
        abnormality_confidence=results.abnormality_detection.confidence,
        triage_priority=results.abnormality_detection.triage_priority,
        key_findings=extract_key_findings(results),
        recommendations=results.recommendations or []
    )
    
    return summary

@router.get("/{job_id}/visualizations/{viz_type}")
async def get_visualization(
    job_id: str,
    viz_type: str,  # eeg_montage, hypnogram, power_spectrum, etc.
    time_start: Optional[float] = None,
    time_end: Optional[float] = None,
    channels: Optional[List[str]] = None,
    current_user: User = Depends(require_permission(Permission.READ_ANALYSIS)),
    db: AsyncSession = Depends(get_db)
):
    """Generate visualization of results."""
    # Validate visualization type
    valid_types = [
        "eeg_montage", "hypnogram", "power_spectrum", 
        "topographic_map", "event_timeline", "feature_importance"
    ]
    
    if viz_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid visualization type. Valid: {valid_types}"
        )
    
    # Get results
    results = await get_results(job_id, current_user=current_user, db=db)
    
    # Generate visualization
    viz_generator = VisualizationGenerator()
    
    if viz_type == "eeg_montage":
        image_bytes = await viz_generator.create_eeg_montage(
            file_id=results.file_id,
            time_start=time_start,
            time_end=time_end,
            channels=channels,
            highlight_events=results.event_detection.events if results.event_detection else None
        )
    elif viz_type == "hypnogram":
        if not results.sleep_analysis:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No sleep analysis results available"
            )
        image_bytes = await viz_generator.create_hypnogram(
            results.sleep_analysis.hypnogram,
            results.sleep_analysis.epoch_timestamps
        )
    # ... other visualization types
    
    return StreamingResponse(
        io.BytesIO(image_bytes),
        media_type="image/png",
        headers={
            "Content-Disposition": f"inline; filename={viz_type}_{job_id}.png",
            "Cache-Control": "max-age=3600"  # Cache for 1 hour
        }
    )
```

### 4. Batch Operations
```python
# src/brain_go_brrr/api/batch.py
router = APIRouter(prefix="/batch", tags=["batch"])

class BatchAnalysisRequest(BaseModel):
    file_ids: List[str]
    analysis_type: AnalysisType = AnalysisType.STANDARD
    priority: str = "normal"
    
    # Batch-specific options
    stop_on_error: bool = False
    parallel_jobs: int = 5
    
    # Common configuration for all files
    common_config: Optional[Dict[str, Any]] = None

class BatchAnalysisResponse(BaseModel):
    batch_id: str
    total_files: int
    job_ids: List[str]
    status: str = "processing"
    links: Dict[str, str]

@router.post(
    "/analyses",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def create_batch_analysis(
    request: BatchAnalysisRequest,
    current_user: User = Depends(require_permission(Permission.WRITE_ANALYSIS)),
    db: AsyncSession = Depends(get_db),
    queue: QueueService = Depends(get_queue)
):
    """Submit multiple EEG files for analysis."""
    # Validate all files exist and are accessible
    accessible_files = []
    for file_id in request.file_ids:
        eeg_file = await get_eeg_file(db, file_id)
        if eeg_file and await can_access_file(current_user, eeg_file):
            accessible_files.append(file_id)
        elif not request.stop_on_error:
            logger.warning(f"Skipping inaccessible file: {file_id}")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot access file: {file_id}"
            )
    
    if not accessible_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No accessible files in batch"
        )
    
    # Create batch job
    batch_id = generate_batch_id()
    job_ids = []
    
    # Create individual jobs
    for file_id in accessible_files:
        job_id = generate_job_id()
        
        analysis_job = AnalysisJob(
            id=job_id,
            file_id=file_id,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            analysis_type=request.analysis_type,
            priority=request.priority,
            batch_id=batch_id,
            configuration=request.common_config or {},
            status="queued"
        )
        
        db.add(analysis_job)
        job_ids.append(job_id)
    
    # Create batch record
    batch = BatchJob(
        id=batch_id,
        user_id=current_user.id,
        total_jobs=len(job_ids),
        job_ids=job_ids,
        configuration=request.dict(),
        status="processing"
    )
    
    db.add(batch)
    await db.commit()
    
    # Submit jobs to queue with rate limiting
    for i, job_id in enumerate(job_ids):
        if i > 0 and i % request.parallel_jobs == 0:
            await asyncio.sleep(1)  # Rate limit
        
        await queue.submit_job(
            job_id=job_id,
            job_type="analysis",
            priority=request.priority,
            payload={
                "file_id": accessible_files[i],
                "batch_id": batch_id
            }
        )
    
    return BatchAnalysisResponse(
        batch_id=batch_id,
        total_files=len(job_ids),
        job_ids=job_ids,
        status="processing",
        links={
            "self": f"/batch/{batch_id}",
            "status": f"/batch/{batch_id}/status",
            "results": f"/batch/{batch_id}/results",
            "cancel": f"/batch/{batch_id}/cancel"
        }
    )

@router.get("/{batch_id}/status", response_model=BatchStatus)
async def get_batch_status(
    batch_id: str,
    current_user: User = Depends(require_permission(Permission.READ_ANALYSIS)),
    db: AsyncSession = Depends(get_db)
):
    """Get status of batch analysis."""
    batch = await get_batch_job(db, batch_id)
    
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Batch not found"
        )
    
    # Get individual job statuses
    job_statuses = await get_job_statuses(db, batch.job_ids)
    
    completed = sum(1 for s in job_statuses if s.status == "completed")
    failed = sum(1 for s in job_statuses if s.status == "failed")
    processing = sum(1 for s in job_statuses if s.status == "processing")
    queued = sum(1 for s in job_statuses if s.status == "queued")
    
    return BatchStatus(
        batch_id=batch_id,
        total_jobs=batch.total_jobs,
        completed=completed,
        failed=failed,
        processing=processing,
        queued=queued,
        progress_percent=int(completed / batch.total_jobs * 100),
        estimated_completion=estimate_batch_completion(job_statuses),
        job_details=[
            {
                "job_id": s.id,
                "file_id": s.file_id,
                "status": s.status,
                "progress": s.progress_percent
            }
            for s in job_statuses
        ]
    )
```

## Error Handling

### Standardized Error Responses
```python
# src/brain_go_brrr/api/errors.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """Handle validation errors with detailed messages."""
    errors = []
    for error in exc.errors():
        field = ".".join(str(x) for x in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details={"errors": errors},
            request_id=request.state.request_id
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail.get("error", "http_error"),
            message=exc.detail.get("message", str(exc.detail)),
            details=exc.detail.get("details"),
            request_id=request.state.request_id
        ).dict()
    )

class BusinessLogicError(HTTPException):
    """Base class for business logic errors."""
    
    def __init__(
        self,
        status_code: int,
        error: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "error": error,
                "message": message,
                "details": details
            }
        )

class InsufficientQuotaError(BusinessLogicError):
    """Raised when user exceeds their quota."""
    
    def __init__(self, quota_type: str, limit: int, used: int):
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            error="insufficient_quota",
            message=f"Exceeded {quota_type} quota",
            details={
                "quota_type": quota_type,
                "limit": limit,
                "used": used,
                "available": max(0, limit - used)
            }
        )
```

## Rate Limiting & Quotas

### API Rate Limiting
```python
# src/brain_go_brrr/api/middleware/rate_limit.py
from fastapi import Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import redis.asyncio as redis

# Configure rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute", "1000 per hour"],
    storage_uri="redis://localhost:6379"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom rate limits for different endpoints
@router.post("/analyses")
@limiter.limit("10 per minute")  # Expensive operation
async def create_analysis(...):
    pass

@router.get("/results/{job_id}")
@limiter.limit("60 per minute")  # Less expensive
async def get_results(...):
    pass

# Organization-based quotas
class QuotaMiddleware:
    """Enforce organization quotas."""
    
    def __init__(self, app):
        self.app = app
        
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Skip for non-authenticated endpoints
            if not request.url.path.startswith("/api/v1"):
                await self.app(scope, receive, send)
                return
            
            # Get user from request
            user = await get_current_user_from_request(request)
            if user:
                # Check quotas
                quota_service = QuotaService()
                
                # Check monthly analysis quota
                if request.method == "POST" and "/analyses" in request.url.path:
                    usage = await quota_service.get_usage(
                        user.organization_id,
                        "analyses",
                        "monthly"
                    )
                    
                    if usage >= user.organization.monthly_analysis_limit:
                        response = JSONResponse(
                            status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            content=ErrorResponse(
                                error="quota_exceeded",
                                message="Monthly analysis quota exceeded",
                                details={
                                    "quota_type": "analyses",
                                    "period": "monthly",
                                    "limit": user.organization.monthly_analysis_limit,
                                    "used": usage,
                                    "reset_date": get_next_month_start()
                                }
                            ).dict()
                        )
                        await response(scope, receive, send)
                        return
                
                # Check storage quota
                if request.method == "POST" and "/eeg/upload" in request.url.path:
                    storage_used = await quota_service.get_storage_usage(
                        user.organization_id
                    )
                    
                    if storage_used >= user.organization.storage_limit_gb * 1024**3:
                        response = JSONResponse(
                            status_code=status.HTTP_402_PAYMENT_REQUIRED,
                            content=ErrorResponse(
                                error="storage_quota_exceeded",
                                message="Storage quota exceeded",
                                details={
                                    "quota_type": "storage",
                                    "limit_gb": user.organization.storage_limit_gb,
                                    "used_gb": storage_used / 1024**3
                                }
                            ).dict()
                        )
                        await response(scope, receive, send)
                        return
            
            await self.app(scope, receive, send)
```

## API Versioning

### Version Management
```python
# src/brain_go_brrr/api/versioning.py
from fastapi import Header, HTTPException

class APIVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"  # Future version

def get_api_version(
    accept: Optional[str] = Header(None),
    x_api_version: Optional[str] = Header(None)
) -> APIVersion:
    """Determine API version from headers."""
    # Priority: X-API-Version header > Accept header > default
    if x_api_version:
        try:
            return APIVersion(x_api_version)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid API version: {x_api_version}"
            )
    
    if accept:
        # Parse Accept header for version
        # application/vnd.brainbrrr.v2+json
        match = re.search(r'vnd\.brainbrrr\.(v\d+)', accept)
        if match:
            try:
                return APIVersion(match.group(1))
            except ValueError:
                pass
    
    # Default to v1
    return APIVersion.V1

# Version-specific routers
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

# Include version-specific endpoints
v1_router.include_router(auth_v1.router)
v1_router.include_router(eeg_v1.router)
v1_router.include_router(analyses_v1.router)

v2_router.include_router(auth_v2.router)
v2_router.include_router(eeg_v2.router)
v2_router.include_router(analyses_v2.router)

app.include_router(v1_router)
app.include_router(v2_router)
```

## API Documentation

### OpenAPI Schema Customization
```python
# src/brain_go_brrr/api/docs.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title="Brain-Go-Brrr API",
        version="1.0.0",
        description="""
        ## Overview
        
        The Brain-Go-Brrr API provides programmatic access to advanced EEG analysis capabilities including:
        
        - **Quality Control**: Automated artifact detection and channel validation
        - **Abnormality Detection**: ML-based identification of abnormal EEG patterns
        - **Sleep Analysis**: Comprehensive sleep staging and metrics
        - **Event Detection**: Identification of epileptiform discharges and other events
        
        ## Authentication
        
        All API requests require authentication via JWT tokens. Obtain tokens via the `/auth/token` endpoint.
        
        ```bash
        curl -X POST https://api.brain-go-brrr.com/v1/auth/token \\
          -H "Content-Type: application/x-www-form-urlencoded" \\
          -d "username=your_username&password=your_password"
        ```
        
        Include the token in subsequent requests:
        
        ```bash
        curl -H "Authorization: Bearer YOUR_TOKEN" \\
          https://api.brain-go-brrr.com/v1/analyses
        ```
        
        ## Rate Limits
        
        - Standard: 100 requests/minute, 1000 requests/hour
        - Analysis creation: 10 requests/minute
        - File uploads: 5 requests/minute
        
        ## Quotas
        
        - Monthly analyses: Based on organization plan
        - Storage: Based on organization plan
        - Concurrent jobs: Based on organization plan
        """,
        routes=app.routes,
        tags=[
            {
                "name": "authentication",
                "description": "Authentication and authorization operations"
            },
            {
                "name": "eeg",
                "description": "EEG file management operations"
            },
            {
                "name": "analyses",
                "description": "Analysis submission and management"
            },
            {
                "name": "results",
                "description": "Analysis results retrieval"
            },
            {
                "name": "batch",
                "description": "Batch processing operations"
            }
        ]
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add webhook documentation
    openapi_schema["webhooks"] = {
        "analysisComplete": {
            "post": {
                "requestBody": {
                    "description": "Analysis completion notification",
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/WebhookAnalysisComplete"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Notification received"}
                }
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

## Webhooks

### Webhook Implementation
```python
# src/brain_go_brrr/api/webhooks.py
import httpx
import hmac
import hashlib

class WebhookService:
    """Manage webhook deliveries."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def send_webhook(
        self,
        url: str,
        event_type: str,
        payload: Dict[str, Any],
        secret: Optional[str] = None
    ):
        """Send webhook with retry logic."""
        headers = {
            "Content-Type": "application/json",
            "X-BrainBrrr-Event": event_type,
            "X-BrainBrrr-Delivery": generate_delivery_id()
        }
        
        # Add signature if secret provided
        if secret:
            body = json.dumps(payload).encode()
            signature = hmac.new(
                secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            headers["X-BrainBrrr-Signature"] = f"sha256={signature}"
        
        # Retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = await self.client.post(
                    url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code < 400:
                    # Success
                    await log_webhook_delivery(
                        url=url,
                        event_type=event_type,
                        status="success",
                        response_code=response.status_code
                    )
                    return
                    
                # Client error - don't retry
                if 400 <= response.status_code < 500:
                    await log_webhook_delivery(
                        url=url,
                        event_type=event_type,
                        status="failed",
                        response_code=response.status_code,
                        error="Client error"
                    )
                    return
                    
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Final attempt failed
                    await log_webhook_delivery(
                        url=url,
                        event_type=event_type,
                        status="failed",
                        error=str(e)
                    )
                    return
                    
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)

# Webhook event schemas
class WebhookAnalysisComplete(BaseModel):
    event: str = "analysis.complete"
    timestamp: datetime
    data: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "event": "analysis.complete",
                "timestamp": "2024-01-15T10:30:00Z",
                "data": {
                    "job_id": "job_12345",
                    "file_id": "file_67890",
                    "status": "completed",
                    "results_summary": {
                        "is_abnormal": True,
                        "confidence": 0.85,
                        "triage_priority": "expedite"
                    }
                }
            }
        }
```

## SDK Generation

### Client SDK Support
```python
# Generate TypeScript SDK
# src/brain_go_brrr/api/sdk/typescript.py

def generate_typescript_sdk():
    """Generate TypeScript SDK from OpenAPI schema."""
    schema = app.openapi()
    
    # Use openapi-typescript-codegen
    sdk_generator = TypeScriptSDKGenerator(schema)
    
    # Generate client code
    client_code = sdk_generator.generate_client()
    
    # Generate type definitions
    types_code = sdk_generator.generate_types()
    
    # Generate documentation
    docs = sdk_generator.generate_docs()
    
    # Package as npm module
    package = {
        "name": "@brain-go-brrr/sdk",
        "version": schema["info"]["version"],
        "main": "dist/index.js",
        "types": "dist/index.d.ts",
        "files": ["dist"],
        "dependencies": {
            "axios": "^1.0.0"
        }
    }
    
    return {
        "src/client.ts": client_code,
        "src/types.ts": types_code,
        "README.md": docs,
        "package.json": json.dumps(package, indent=2)
    }

# Python SDK generation
def generate_python_sdk():
    """Generate Python SDK from OpenAPI schema."""
    schema = app.openapi()
    
    # Use openapi-python-client
    sdk_generator = PythonSDKGenerator(schema)
    
    return sdk_generator.generate_package()
```

## Testing

### API Testing Suite
```python
# tests/api/test_endpoints.py
import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient

@pytest.mark.asyncio
class TestAPIEndpoints:
    """Comprehensive API testing."""
    
    async def test_authentication_flow(self, client: AsyncClient):
        """Test complete authentication flow."""
        # Register user
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "organization": "Test Org"
            }
        )
        assert response.status_code == 201
        
        # Login
        response = await client.post(
            "/api/v1/auth/token",
            data={
                "username": "test@example.com",
                "password": "SecurePass123!"
            }
        )
        assert response.status_code == 200
        token = response.json()["access_token"]
        
        # Use token
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get("/api/v1/user/profile", headers=headers)
        assert response.status_code == 200
        
    async def test_analysis_workflow(self, client: AsyncClient, auth_headers: dict):
        """Test complete analysis workflow."""
        # Upload file
        with open("test_data/sample.edf", "rb") as f:
            response = await client.post(
                "/api/v1/eeg/upload",
                files={"file": ("sample.edf", f, "application/octet-stream")},
                headers=auth_headers
            )
        assert response.status_code == 201
        file_id = response.json()["file_id"]
        
        # Submit analysis
        response = await client.post(
            "/api/v1/analyses",
            json={
                "file_id": file_id,
                "analysis_type": "comprehensive"
            },
            headers=auth_headers
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        
        # Check status
        response = await client.get(
            f"/api/v1/analyses/{job_id}/status",
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["status"] in ["queued", "processing", "completed"]
        
        # Wait for completion (in real test, use mock)
        await wait_for_job_completion(client, job_id, auth_headers)
        
        # Get results
        response = await client.get(
            f"/api/v1/results/{job_id}",
            headers=auth_headers
        )
        assert response.status_code == 200
        results = response.json()
        assert "abnormality_detection" in results
        assert "quality_report" in results
```

## Conclusion

This API design provides a robust, scalable, and developer-friendly interface for the Brain-Go-Brrr EEG analysis platform. The design prioritizes security, reliability, and ease of use while maintaining compliance with medical data regulations.