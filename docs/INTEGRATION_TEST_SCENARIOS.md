# Integration Test Scenarios for Brain-Go-Brrr

## Executive Summary

This document defines comprehensive integration test scenarios for the Brain-Go-Brrr EEG analysis pipeline. These tests ensure that all components work together seamlessly to deliver accurate, reliable results in real-world scenarios.

## Integration Testing Philosophy

### Key Principles
1. **End-to-End Validation**: Test complete workflows from input to output
2. **Real Data**: Use actual EEG recordings when possible
3. **Error Propagation**: Verify error handling across component boundaries
4. **Performance Constraints**: Ensure integrated system meets performance targets
5. **Clinical Relevance**: Test scenarios that matter in medical settings

## Test Environment Setup

### Infrastructure Requirements
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  postgres:
    image: timescale/timescaledb:2.11.0-pg15
    environment:
      POSTGRES_DB: braintest
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass
    volumes:
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
  
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass testpass
  
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - ./test-data:/data
  
  api:
    build: .
    environment:
      DATABASE_URL: postgresql://testuser:testpass@postgres:5432/braintest
      REDIS_URL: redis://:testpass@redis:6379/0
      S3_ENDPOINT: http://minio:9000
      MODEL_PATH: /models
    volumes:
      - ./models:/models
    depends_on:
      - postgres
      - redis
      - minio
```

## Component Integration Scenarios

### 1. Data Flow Integration Tests

#### Test 1.1: EDF Upload to Analysis Complete
```python
# tests/integration/test_data_flow.py
import pytest
from pathlib import Path
import asyncio
from httpx import AsyncClient

@pytest.mark.integration
class TestDataFlowIntegration:
    """Test complete data flow from upload to results."""
    
    async def test_edf_upload_to_analysis_complete(
        self,
        test_client: AsyncClient,
        sample_edf_file: Path,
        redis_client
    ):
        """Test full pipeline from EDF upload to analysis results."""
        # Step 1: Upload EDF file
        with open(sample_edf_file, 'rb') as f:
            response = await test_client.post(
                "/api/v1/eeg/upload",
                files={"file": ("test.edf", f, "application/octet-stream")}
            )
        
        assert response.status_code == 200
        file_id = response.json()["file_id"]
        
        # Step 2: Submit for analysis
        analysis_response = await test_client.post(
            "/api/v1/eeg/analyze",
            json={
                "file_id": file_id,
                "analysis_type": "comprehensive",
                "options": {
                    "quality_control": True,
                    "abnormality_detection": True,
                    "sleep_analysis": True,
                    "event_detection": True
                }
            }
        )
        
        assert analysis_response.status_code == 202
        job_id = analysis_response.json()["job_id"]
        
        # Step 3: Wait for completion
        max_wait = 120  # 2 minutes
        start_time = asyncio.get_event_loop().time()
        
        while True:
            status_response = await test_client.get(f"/api/v1/jobs/{job_id}")
            status = status_response.json()["status"]
            
            if status == "completed":
                break
            elif status == "failed":
                pytest.fail(f"Job failed: {status_response.json()['error']}")
            
            if asyncio.get_event_loop().time() - start_time > max_wait:
                pytest.fail("Analysis timed out")
            
            await asyncio.sleep(2)
        
        # Step 4: Retrieve results
        results_response = await test_client.get(f"/api/v1/results/{job_id}")
        assert results_response.status_code == 200
        
        results = results_response.json()
        
        # Validate results structure
        assert "quality_report" in results
        assert "abnormality_detection" in results
        assert "sleep_analysis" in results
        assert "event_detection" in results
        
        # Validate quality report
        qc = results["quality_report"]
        assert "bad_channels" in qc
        assert "signal_quality_score" in qc
        assert qc["signal_quality_score"] >= 0 and qc["signal_quality_score"] <= 1
        
        # Validate abnormality detection
        abnormal = results["abnormality_detection"]
        assert "is_abnormal" in abnormal
        assert "confidence" in abnormal
        assert "triage_priority" in abnormal
        assert abnormal["triage_priority"] in ["routine", "expedite", "urgent"]
        
        # Validate sleep analysis (if applicable)
        if results["sleep_analysis"]:
            sleep = results["sleep_analysis"]
            assert "hypnogram" in sleep
            assert "sleep_efficiency" in sleep
            assert "sleep_stages" in sleep
        
        # Step 5: Verify data persistence
        # Check S3
        s3_key = f"results/{job_id}/complete.json"
        assert await s3_client.object_exists("results", s3_key)
        
        # Check database
        db_result = await db.fetch_one(
            "SELECT * FROM analysis_results WHERE job_id = :job_id",
            {"job_id": job_id}
        )
        assert db_result is not None
        assert db_result["status"] == "completed"
```

#### Test 1.2: Multi-File Batch Processing
```python
async def test_batch_processing_integration(
    self,
    test_client: AsyncClient,
    edf_files: List[Path]
):
    """Test processing multiple files in batch."""
    # Upload all files
    file_ids = []
    for edf_file in edf_files[:10]:  # Test with 10 files
        with open(edf_file, 'rb') as f:
            response = await test_client.post(
                "/api/v1/eeg/upload",
                files={"file": (edf_file.name, f, "application/octet-stream")}
            )
        file_ids.append(response.json()["file_id"])
    
    # Submit batch analysis
    batch_response = await test_client.post(
        "/api/v1/batch/analyze",
        json={
            "file_ids": file_ids,
            "analysis_type": "screening",
            "priority": "normal"
        }
    )
    
    assert batch_response.status_code == 202
    batch_id = batch_response.json()["batch_id"]
    
    # Monitor batch progress
    completed = 0
    failed = 0
    
    while completed + failed < len(file_ids):
        status_response = await test_client.get(f"/api/v1/batch/{batch_id}/status")
        batch_status = status_response.json()
        
        completed = batch_status["completed"]
        failed = batch_status["failed"]
        
        # Verify progress updates
        assert batch_status["total"] == len(file_ids)
        assert batch_status["progress"] >= 0 and batch_status["progress"] <= 100
        
        await asyncio.sleep(5)
    
    # Verify all completed successfully
    assert completed == len(file_ids)
    assert failed == 0
    
    # Download batch results
    results_response = await test_client.get(f"/api/v1/batch/{batch_id}/results")
    assert results_response.status_code == 200
    
    results = results_response.json()
    assert len(results["analyses"]) == len(file_ids)
```

### 2. Model Integration Tests

#### Test 2.1: EEGPT + AutoReject Integration
```python
# tests/integration/test_model_integration.py

@pytest.mark.integration
class TestModelIntegration:
    """Test model component integration."""
    
    def test_eegpt_autoreject_integration(self, sample_raw_eeg):
        """Test EEGPT with AutoReject preprocessing."""
        # Initialize components
        autoreject = AutoRejectWrapper()
        eegpt = EEGPTModel.from_checkpoint(CHECKPOINT_PATH)
        
        # Original feature extraction
        original_features = []
        windows = sliding_window(sample_raw_eeg, window_size=8, stride=4)
        for window in windows:
            feat = eegpt.extract_features(window)
            original_features.append(feat)
        
        # Process with AutoReject
        cleaned_raw, bad_channels = autoreject.process(sample_raw_eeg)
        
        # Feature extraction on cleaned data
        cleaned_features = []
        cleaned_windows = sliding_window(cleaned_raw, window_size=8, stride=4)
        for window in cleaned_windows:
            feat = eegpt.extract_features(window)
            cleaned_features.append(feat)
        
        # Compare features
        # Cleaned features should be different but correlated
        original_mean = torch.stack(original_features).mean(dim=0)
        cleaned_mean = torch.stack(cleaned_features).mean(dim=0)
        
        correlation = torch.corrcoef(
            torch.stack([original_mean, cleaned_mean])
        )[0, 1]
        
        # High correlation but not identical
        assert 0.7 < correlation < 0.99
        
        # Verify bad channels were handled
        if bad_channels:
            assert len(cleaned_raw.ch_names) == len(sample_raw_eeg.ch_names)
            # Bad channels should be interpolated, not removed
```

#### Test 2.2: Hierarchical Detection Pipeline
```python
def test_hierarchical_detection_pipeline(self, abnormal_eeg_samples):
    """Test hierarchical abnormal → event detection."""
    pipeline = HierarchicalDetectionPipeline()
    
    for eeg_sample in abnormal_eeg_samples:
        # Step 1: Quality control
        qc_result = pipeline.quality_control(eeg_sample)
        assert qc_result.is_acceptable
        
        # Step 2: Abnormality detection
        abnormal_result = pipeline.detect_abnormality(qc_result.cleaned_data)
        
        # Step 3: Conditional event detection
        if abnormal_result.is_abnormal:
            event_result = pipeline.detect_events(qc_result.cleaned_data)
            assert event_result is not None
            
            # Verify event types
            event_types = {e.event_type for e in event_result.events}
            valid_types = {
                "epileptiform", "PLED", "GPED", 
                "seizure", "artifact", "burst_suppression"
            }
            assert event_types.issubset(valid_types)
        else:
            # No event detection for normal EEG
            event_result = pipeline.detect_events(qc_result.cleaned_data)
            assert event_result is None
```

### 3. Database Integration Tests

#### Test 3.1: Transaction Integrity
```python
# tests/integration/test_database_integration.py

@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database transaction integrity."""
    
    async def test_analysis_transaction_integrity(self, db_session):
        """Test atomic transaction for analysis results."""
        job_id = "test-job-123"
        
        try:
            async with db_session.begin():
                # Insert analysis job
                await db_session.execute(
                    """
                    INSERT INTO analysis_jobs (job_id, status, created_at)
                    VALUES (:job_id, :status, :created_at)
                    """,
                    {
                        "job_id": job_id,
                        "status": "processing",
                        "created_at": datetime.utcnow()
                    }
                )
                
                # Simulate processing
                results = {
                    "abnormality_score": 0.85,
                    "confidence": 0.92
                }
                
                # Update with results
                await db_session.execute(
                    """
                    UPDATE analysis_jobs 
                    SET status = :status, 
                        results = :results,
                        completed_at = :completed_at
                    WHERE job_id = :job_id
                    """,
                    {
                        "job_id": job_id,
                        "status": "completed",
                        "results": json.dumps(results),
                        "completed_at": datetime.utcnow()
                    }
                )
                
                # Insert metrics
                await db_session.execute(
                    """
                    INSERT INTO analysis_metrics 
                    (job_id, processing_time_ms, model_inference_time_ms)
                    VALUES (:job_id, :processing_time, :inference_time)
                    """,
                    {
                        "job_id": job_id,
                        "processing_time": 45000,
                        "inference_time": 12000
                    }
                )
                
                # Simulate error to test rollback
                if random.random() < 0.1:  # 10% failure rate
                    raise Exception("Simulated processing error")
                
        except Exception as e:
            # Verify rollback
            result = await db_session.fetch_one(
                "SELECT * FROM analysis_jobs WHERE job_id = :job_id",
                {"job_id": job_id}
            )
            assert result is None or result["status"] != "completed"
            raise
        
        # Verify all committed
        job = await db_session.fetch_one(
            "SELECT * FROM analysis_jobs WHERE job_id = :job_id",
            {"job_id": job_id}
        )
        assert job["status"] == "completed"
        
        metrics = await db_session.fetch_one(
            "SELECT * FROM analysis_metrics WHERE job_id = :job_id",
            {"job_id": job_id}
        )
        assert metrics is not None
```

#### Test 3.2: TimescaleDB Performance
```python
async def test_timescale_performance(self, db_session):
    """Test TimescaleDB hypertable performance."""
    # Insert time-series metrics
    start_time = datetime.utcnow() - timedelta(hours=24)
    
    # Generate 24 hours of metrics (1 per minute)
    metrics = []
    for i in range(24 * 60):
        timestamp = start_time + timedelta(minutes=i)
        metrics.append({
            "timestamp": timestamp,
            "active_jobs": random.randint(0, 50),
            "cpu_usage": random.uniform(0, 100),
            "memory_usage": random.uniform(0, 16384),
            "gpu_usage": random.uniform(0, 100)
        })
    
    # Bulk insert
    start = time.time()
    await db_session.execute_many(
        """
        INSERT INTO system_metrics 
        (timestamp, active_jobs, cpu_usage, memory_usage, gpu_usage)
        VALUES (:timestamp, :active_jobs, :cpu_usage, :memory_usage, :gpu_usage)
        """,
        metrics
    )
    insert_time = time.time() - start
    
    # Should handle 1440 inserts quickly
    assert insert_time < 1.0  # Less than 1 second
    
    # Test time-range queries
    start = time.time()
    hourly_avg = await db_session.fetch_all(
        """
        SELECT 
            time_bucket('1 hour', timestamp) as hour,
            avg(cpu_usage) as avg_cpu,
            avg(memory_usage) as avg_memory,
            max(active_jobs) as max_jobs
        FROM system_metrics
        WHERE timestamp >= :start_time
        GROUP BY hour
        ORDER BY hour
        """,
        {"start_time": start_time}
    )
    query_time = time.time() - start
    
    assert len(hourly_avg) == 24
    assert query_time < 0.1  # Less than 100ms
```

### 4. Cache Integration Tests

#### Test 4.1: Multi-Level Cache Integration
```python
# tests/integration/test_cache_integration.py

@pytest.mark.integration
class TestCacheIntegration:
    """Test multi-level cache integration."""
    
    async def test_cache_hierarchy(self, redis_client, s3_client):
        """Test L1 (Redis) and L2 (S3) cache integration."""
        cache = MultiLevelCache(
            l1_cache=RedisCache(redis_client),
            l2_cache=S3Cache(s3_client, bucket="cache")
        )
        
        # Generate test data
        feature_data = torch.randn(768).numpy()
        cache_key = "features:test-file:window-0"
        
        # Initial miss
        result = await cache.get(cache_key)
        assert result is None
        
        # Store in cache
        await cache.set(cache_key, feature_data, ttl=3600)
        
        # L1 hit
        l1_result = await cache.l1_cache.get(cache_key)
        assert l1_result is not None
        assert np.allclose(l1_result, feature_data)
        
        # Simulate L1 eviction
        await cache.l1_cache.delete(cache_key)
        
        # L2 hit with L1 repopulation
        result = await cache.get(cache_key)
        assert result is not None
        assert np.allclose(result, feature_data)
        
        # Verify L1 was repopulated
        l1_result = await cache.l1_cache.get(cache_key)
        assert l1_result is not None
```

#### Test 4.2: Cache Invalidation
```python
async def test_cache_invalidation_cascade(self, cache_system):
    """Test cache invalidation across components."""
    file_id = "test-file-123"
    
    # Populate caches
    await cache_system.set(f"raw:{file_id}", "raw_data")
    await cache_system.set(f"features:{file_id}:0", "features_0")
    await cache_system.set(f"features:{file_id}:1", "features_1")
    await cache_system.set(f"results:{file_id}", "results")
    
    # Invalidate file (should cascade)
    await cache_system.invalidate_file(file_id)
    
    # Verify all related entries removed
    assert await cache_system.get(f"raw:{file_id}") is None
    assert await cache_system.get(f"features:{file_id}:0") is None
    assert await cache_system.get(f"features:{file_id}:1") is None
    assert await cache_system.get(f"results:{file_id}") is None
```

### 5. Error Handling Integration Tests

#### Test 5.1: Cascading Failure Handling
```python
# tests/integration/test_error_handling.py

@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across components."""
    
    async def test_model_loading_failure_recovery(self, test_client):
        """Test recovery from model loading failures."""
        # Simulate model corruption
        with mock.patch('brain_go_brrr.models.eegpt.load_checkpoint') as mock_load:
            mock_load.side_effect = RuntimeError("Corrupted model file")
            
            # Try analysis
            response = await test_client.post(
                "/api/v1/eeg/analyze",
                json={"file_id": "test-file", "analysis_type": "quick"}
            )
            
            # Should fail gracefully
            assert response.status_code == 503
            error = response.json()
            assert error["error"] == "service_unavailable"
            assert "model initialization failed" in error["message"].lower()
        
        # Verify automatic recovery attempt
        with mock.patch('brain_go_brrr.models.eegpt.download_model') as mock_download:
            mock_download.return_value = True
            
            # Retry should trigger re-download
            response = await test_client.post(
                "/api/v1/eeg/analyze",
                json={"file_id": "test-file", "analysis_type": "quick"}
            )
            
            mock_download.assert_called_once()
```

#### Test 5.2: Partial Failure Recovery
```python
async def test_partial_analysis_recovery(self, pipeline, problematic_eeg):
    """Test recovery from partial analysis failures."""
    results = await pipeline.analyze_with_recovery(problematic_eeg)
    
    # Quality control should always complete
    assert results["quality_control"]["status"] == "completed"
    
    # Abnormality detection might fail on bad data
    if results["abnormality_detection"]["status"] == "failed":
        assert "error" in results["abnormality_detection"]
        assert results["abnormality_detection"]["fallback_result"] is not None
        
        # Fallback should be conservative
        fallback = results["abnormality_detection"]["fallback_result"]
        assert fallback["is_abnormal"] == True  # Err on side of caution
        assert fallback["confidence"] < 0.5  # Low confidence
        assert fallback["triage_priority"] == "expedite"  # Not urgent, but not routine
    
    # Downstream analyses should handle upstream failures
    if results["event_detection"]["status"] == "skipped":
        assert results["event_detection"]["reason"] == "upstream_failure"
```

### 6. Performance Integration Tests

#### Test 6.1: Load Testing
```python
# tests/integration/test_performance_integration.py
import locust

class EEGAnalysisUser(locust.HttpUser):
    """Locust user for load testing."""
    wait_time = locust.between(1, 5)
    
    def on_start(self):
        """Upload test files on start."""
        self.file_ids = []
        for i in range(10):
            with open(f"test_data/eeg_{i}.edf", "rb") as f:
                response = self.client.post(
                    "/api/v1/eeg/upload",
                    files={"file": f}
                )
                self.file_ids.append(response.json()["file_id"])
    
    @locust.task(3)
    def analyze_quick(self):
        """Quick analysis - most common."""
        file_id = random.choice(self.file_ids)
        with self.client.post(
            "/api/v1/eeg/analyze",
            json={
                "file_id": file_id,
                "analysis_type": "quick"
            },
            catch_response=True
        ) as response:
            if response.elapsed.total_seconds() > 0.1:
                response.failure("Analysis took too long")
    
    @locust.task(1)
    def analyze_comprehensive(self):
        """Comprehensive analysis - less common."""
        file_id = random.choice(self.file_ids)
        self.client.post(
            "/api/v1/eeg/analyze",
            json={
                "file_id": file_id,
                "analysis_type": "comprehensive"
            }
        )
    
    @locust.task(2)
    def check_status(self):
        """Status checks."""
        job_id = f"job-{random.randint(1000, 9999)}"
        self.client.get(f"/api/v1/jobs/{job_id}")
```

#### Test 6.2: Memory Leak Detection
```python
@pytest.mark.integration
@pytest.mark.slow
async def test_memory_leak_under_load(test_client, monitoring):
    """Test for memory leaks during sustained load."""
    initial_memory = monitoring.get_memory_usage()
    
    # Process 100 files
    for i in range(100):
        # Upload and analyze
        with open(f"test_data/eeg_{i % 10}.edf", "rb") as f:
            upload_response = await test_client.post(
                "/api/v1/eeg/upload",
                files={"file": f}
            )
        
        file_id = upload_response.json()["file_id"]
        
        analyze_response = await test_client.post(
            "/api/v1/eeg/analyze",
            json={
                "file_id": file_id,
                "analysis_type": "quick"
            }
        )
        
        job_id = analyze_response.json()["job_id"]
        
        # Wait for completion
        await wait_for_job_completion(test_client, job_id)
        
        # Check memory every 10 iterations
        if i % 10 == 0:
            current_memory = monitoring.get_memory_usage()
            memory_increase = current_memory - initial_memory
            
            # Allow some increase, but should stabilize
            assert memory_increase < 500 * (i // 10 + 1)  # Max 500MB per 10 files
    
    # Force garbage collection
    import gc
    gc.collect()
    await asyncio.sleep(5)
    
    # Final memory should be close to initial
    final_memory = monitoring.get_memory_usage()
    total_increase = final_memory - initial_memory
    assert total_increase < 1000  # Less than 1GB increase
```

### 7. Security Integration Tests

#### Test 7.1: Authentication Flow
```python
# tests/integration/test_security_integration.py

@pytest.mark.integration
class TestSecurityIntegration:
    """Test security across components."""
    
    async def test_jwt_authentication_flow(self, test_client):
        """Test JWT authentication through full stack."""
        # Get token
        token_response = await test_client.post(
            "/api/v1/auth/token",
            data={
                "username": "test_user",
                "password": "test_password"
            }
        )
        assert token_response.status_code == 200
        token = token_response.json()["access_token"]
        
        # Use token for authenticated request
        headers = {"Authorization": f"Bearer {token}"}
        
        # Should succeed with valid token
        response = await test_client.get(
            "/api/v1/user/analyses",
            headers=headers
        )
        assert response.status_code == 200
        
        # Should fail with invalid token
        bad_headers = {"Authorization": "Bearer invalid_token"}
        response = await test_client.get(
            "/api/v1/user/analyses",
            headers=bad_headers
        )
        assert response.status_code == 401
```

#### Test 7.2: Data Encryption
```python
async def test_data_encryption_pipeline(self, secure_storage):
    """Test encryption through storage pipeline."""
    sensitive_data = {
        "patient_id": "12345",
        "diagnosis": "epilepsy",
        "medications": ["levetiracetam", "lamotrigine"]
    }
    
    # Store encrypted
    key = "patient:12345:medical"
    await secure_storage.store_encrypted(key, sensitive_data)
    
    # Verify encrypted in storage
    raw_data = await secure_storage.get_raw(key)
    assert raw_data != json.dumps(sensitive_data)  # Should be encrypted
    assert "patient_id" not in str(raw_data)  # No plaintext
    
    # Retrieve decrypted
    decrypted = await secure_storage.get_encrypted(key)
    assert decrypted == sensitive_data
```

### 8. Monitoring Integration Tests

#### Test 8.1: Metrics Collection
```python
# tests/integration/test_monitoring_integration.py

@pytest.mark.integration
async def test_metrics_collection_pipeline(monitoring_client, test_client):
    """Test metrics collection across components."""
    # Clear metrics
    monitoring_client.clear_metrics()
    
    # Perform operations
    with open("test_data/sample.edf", "rb") as f:
        upload_response = await test_client.post(
            "/api/v1/eeg/upload",
            files={"file": f}
        )
    
    analyze_response = await test_client.post(
        "/api/v1/eeg/analyze",
        json={
            "file_id": upload_response.json()["file_id"],
            "analysis_type": "comprehensive"
        }
    )
    
    job_id = analyze_response.json()["job_id"]
    await wait_for_job_completion(test_client, job_id)
    
    # Verify metrics collected
    metrics = monitoring_client.get_metrics()
    
    # API metrics
    assert "http_requests_total" in metrics
    assert metrics["http_requests_total"]["POST"]["/api/v1/eeg/upload"] >= 1
    assert metrics["http_requests_total"]["POST"]["/api/v1/eeg/analyze"] >= 1
    
    # Processing metrics
    assert "eeg_processing_duration_seconds" in metrics
    assert len(metrics["eeg_processing_duration_seconds"]) > 0
    
    # Model metrics
    assert "model_inference_duration_seconds" in metrics
    assert "eegpt" in metrics["model_inference_duration_seconds"]
    
    # Cache metrics
    assert "cache_hits_total" in metrics
    assert "cache_misses_total" in metrics
```

### 9. Disaster Recovery Tests

#### Test 9.1: Service Recovery
```python
# tests/integration/test_disaster_recovery.py

@pytest.mark.integration
class TestDisasterRecovery:
    """Test disaster recovery scenarios."""
    
    async def test_database_failover(self, test_env):
        """Test database failover handling."""
        # Start analysis
        job_id = await submit_analysis(test_env.api_client)
        
        # Simulate primary database failure
        await test_env.stop_service("postgres-primary")
        
        # System should failover to replica
        await asyncio.sleep(5)  # Allow failover
        
        # Should still be able to check status
        response = await test_env.api_client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        
        # New writes should go to replica (promoted to primary)
        new_job_id = await submit_analysis(test_env.api_client)
        assert new_job_id is not None
        
        # Restore primary
        await test_env.start_service("postgres-primary")
        await asyncio.sleep(10)  # Allow replication sync
        
        # Verify data consistency
        primary_data = await test_env.query_database(
            "postgres-primary",
            f"SELECT * FROM analysis_jobs WHERE job_id = '{new_job_id}'"
        )
        replica_data = await test_env.query_database(
            "postgres-replica", 
            f"SELECT * FROM analysis_jobs WHERE job_id = '{new_job_id}'"
        )
        assert primary_data == replica_data
```

#### Test 9.2: Queue Recovery
```python
async def test_queue_recovery_after_crash(test_env):
    """Test job queue recovery after crash."""
    # Submit multiple jobs
    job_ids = []
    for i in range(10):
        job_id = await submit_analysis(test_env.api_client)
        job_ids.append(job_id)
    
    # Simulate worker crash during processing
    await asyncio.sleep(2)  # Let some jobs start
    await test_env.stop_service("celery-worker")
    
    # Check job states
    states_before = {}
    for job_id in job_ids:
        response = await test_env.api_client.get(f"/api/v1/jobs/{job_id}")
        states_before[job_id] = response.json()["status"]
    
    # Restart worker
    await test_env.start_service("celery-worker")
    await asyncio.sleep(5)
    
    # Verify jobs resume
    max_wait = 120
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        all_completed = True
        for job_id in job_ids:
            response = await test_env.api_client.get(f"/api/v1/jobs/{job_id}")
            status = response.json()["status"]
            if status not in ["completed", "failed"]:
                all_completed = False
                break
        
        if all_completed:
            break
        await asyncio.sleep(5)
    
    # Verify all jobs completed
    for job_id in job_ids:
        response = await test_env.api_client.get(f"/api/v1/jobs/{job_id}")
        final_status = response.json()["status"]
        assert final_status in ["completed", "failed"]
        
        # If was processing before crash, should have been retried
        if states_before[job_id] == "processing":
            assert response.json().get("retry_count", 0) > 0
```

## Integration Test Execution

### Test Environment Setup
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
uv run pytest tests/integration/ -v --maxfail=1

# Run specific scenario
uv run pytest tests/integration/test_data_flow.py::TestDataFlowIntegration::test_edf_upload_to_analysis_complete -xvs

# Run with coverage
uv run pytest tests/integration/ --cov=brain_go_brrr --cov-report=html

# Load testing
locust -f tests/integration/test_performance_integration.py --host=http://localhost:8000
```

### CI/CD Integration
```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on:
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM

jobs:
  integration:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: timescale/timescaledb:2.11.0-pg15
        env:
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
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
    
    - name: Download test data
      run: |
        wget -O test_data.tar.gz https://example.com/test-eeg-data.tar.gz
        tar -xzf test_data.tar.gz
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:testpass@localhost:5432/test
        REDIS_URL: redis://localhost:6379/0
      run: |
        uv run pytest tests/integration/ -v --tb=short
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: integration-test-results
        path: |
          test-results.xml
          htmlcov/
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Data Cleanup**: Always clean up test data
3. **Timeouts**: Set reasonable timeouts for async operations
4. **Mocking**: Mock external services when appropriate
5. **Monitoring**: Track test execution time and flakiness
6. **Documentation**: Document complex test scenarios

## Conclusion

These comprehensive integration test scenarios ensure that Brain-Go-Brrr functions reliably as a complete system, handling real-world complexities and edge cases while maintaining performance and accuracy requirements.