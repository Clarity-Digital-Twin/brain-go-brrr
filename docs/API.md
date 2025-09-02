# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

⚠️ **Currently no authentication implemented** - API is open access

## Pathway Selection

The API provides multiple processing pathways based on your needs:

- **Quality Control**: `/api/v1/eeg/analyze` - Synchronous bad channel detection
- **EEGPT Analysis**: `/api/v1/eeg/eegpt/analyze` - Requires 19+ channels @ 256Hz  
- **Sleep Staging**: `/api/v1/eeg/sleep/analyze` - Works with ANY channel count (1-100+)

**Note**: YASA is NOT limited to 2-channel data. It works with any channel count and automatically selects the best central channel (C3/C4 preferred) for optimal accuracy.

## Endpoints

### Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-09-02T00:00:00Z"
}
```

### Quality Control Analysis (Synchronous)

```http
POST /api/v1/eeg/analyze
```

**Request**: Multipart form data
- `edf_file`: EDF/BDF file upload

**Response** (QCResponse):
```json
{
  "bad_channels": ["F7", "T4"],
  "quality_metrics": {
    "snr": 12.5,
    "artifacts": {
      "eye_blinks": 23,
      "muscle": 45,
      "heartbeat": 12
    },
    "usable_percentage": 87.5
  },
  "processing_time": 2.3,
  "timestamp": "2025-09-02T00:00:00Z"
}
```

### EEGPT Probe Analysis

```http
POST /api/v1/eeg/eegpt/analyze
```

**Request**: Multipart form data
- `edf_file`: EDF/BDF file upload
- `analysis_type`: "abnormality_probe" | "sleep_probe" | "motor_imagery_probe"

**Response** (EEGPTAnalysisResponse):
```json
{
  "prediction": "normal",
  "confidence": 0.92,
  "probabilities": {
    "normal": 0.92,
    "abnormal": 0.08
  },
  "probe_type": "abnormality_probe",
  "features_shape": [1, 512],
  "processing_time": 3.1
}
```

### EEGPT Batch Analysis

```http
POST /api/v1/eeg/eegpt/analyze/batch
```

**Request**: Multipart form data
- `edf_file`: EDF/BDF file upload
- `batch_size`: Number of windows to process (default: 32)
- `analysis_type`: Analysis type (default: "abnormality")

**Response**:
```json
{
  "windows_processed": 32,
  "predictions": [
    {"window": 0, "prediction": "normal", "confidence": 0.92},
    {"window": 1, "prediction": "normal", "confidence": 0.88}
  ],
  "average_confidence": 0.89,
  "processing_time": 12.5
}
```

### Sleep Analysis (Asynchronous)

```http
POST /api/v1/eeg/sleep/analyze
```

**Request**: Multipart form data
- `edf_file`: EDF/BDF file upload

**Response** (JobResponse - 202 Accepted):
```json
{
  "job_id": "uuid-string",
  "status": "queued",
  "created_at": "2025-09-02T00:00:00Z",
  "message": "Sleep analysis job queued"
}
```

### Get Job Results

```http
GET /api/v1/jobs/{job_id}
```

**Response** (Success):
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "result": {
    "hypnogram": [0, 0, 1, 2, 2, 3, 3, 2, 4],
    "sleep_stages": {
      "W": 15.2,
      "N1": 7.8,
      "N2": 45.3,
      "N3": 20.1,
      "REM": 11.6
    },
    "sleep_efficiency": 84.8,
    "total_sleep_time_minutes": 423.5
  },
  "completed_at": "2025-09-02T00:01:30Z"
}
```

### Queue Status

```http
GET /api/v1/queue/status
```

**Response**:
```json
{
  "total_jobs": 5,
  "queued": 2,
  "processing": 1,
  "completed": 2,
  "failed": 0
}
```

### Upload EEG File (Deprecated - use specific endpoints above)

```http
POST /api/v1/eeg/upload
```

**Request**: Multipart form data with EDF/BDF file

**Response**:
```json
{
  "file_id": "uuid-string",
  "filename": "recording.edf",
  "size_mb": 125.3,
  "duration_hours": 8.5,
  "channels": 19
}
```

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 202 | Accepted (async job queued) |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

## Rate Limiting

Currently no rate limiting implemented.

## Caching

Redis caching enabled with configurable TTL (default: 2 hours) for analysis results.

## WebSocket Support

Not implemented.

## SDK Example

Python SDK example:

```python
import requests

# Quality Control (synchronous)
with open("data/sample.edf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/eeg/analyze",
        files={"edf_file": f}
    )
    qc_results = response.json()
    print(f"Bad channels: {qc_results['bad_channels']}")

# EEGPT Analysis (synchronous)
with open("data/sample.edf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/eeg/eegpt/analyze",
        files={"edf_file": f},
        data={"analysis_type": "abnormality_probe"}
    )
    eegpt_results = response.json()
    print(f"Prediction: {eegpt_results['prediction']}")

# Sleep Analysis (asynchronous)
with open("data/sample.edf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/eeg/sleep/analyze",
        files={"edf_file": f}
    )
    job = response.json()
    job_id = job["job_id"]
    
    # Poll for results
    import time
    while True:
        result = requests.get(f"http://localhost:8000/api/v1/jobs/{job_id}")
        if result.json()["status"] == "completed":
            print(result.json()["result"])
            break
        time.sleep(5)
```

## OpenAPI Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

All errors return JSON with structure:

```json
{
  "detail": "Error message here"
}
```

Or for validation errors:

```json
{
  "detail": [
    {
      "loc": ["body", "edf_file"],
      "msg": "File must be EDF or BDF format",
      "type": "value_error"
    }
  ]
}
```

## Performance

- Average response time: <100ms (cached), <5s (QC), <30s (EEGPT)
- Max file size: 2GB
- Concurrent requests: 50
- Memory limit per request: 4GB

## CORS

CORS enabled for all origins in development. Production will restrict to specific domains.

## Monitoring

Health metrics exposed at `/metrics` (Prometheus format) - not yet implemented.