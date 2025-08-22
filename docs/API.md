# API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

⚠️ **Currently no authentication implemented** - API is open access

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
  "timestamp": "2025-08-22T00:00:00Z"
}
```

### EEG Analysis

```http
POST /api/v1/eeg/analyze
```

**Request Body**:
```json
{
  "file_path": "path/to/eeg.edf",
  "analysis_types": ["sleep", "quality", "abnormality"],
  "options": {
    "window_size": 4.0,
    "overlap": 0.5
  }
}
```

**Response**:
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "created_at": "2025-08-22T00:00:00Z"
}
```

### Get Results

```http
GET /api/v1/eeg/results/{job_id}
```

**Response** (Success):
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "results": {
    "quality": {
      "bad_channels": ["F7", "T4"],
      "artifacts": {
        "eye_blinks": 23,
        "muscle": 45,
        "heartbeat": 12
      },
      "usable_percentage": 87.5
    },
    "sleep": {
      "stages": {
        "W": 15.2,
        "N1": 7.8,
        "N2": 45.3,
        "N3": 20.1,
        "REM": 11.6
      },
      "efficiency": 84.8,
      "total_sleep_time": 423.5
    },
    "abnormality": {
      "classification": "normal",
      "confidence": 0.92,
      "flag": "ROUTINE"
    }
  }
}
```

### Upload EEG File

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
| 201 | Created |
| 400 | Bad Request (invalid parameters) |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

## Rate Limiting

Currently no rate limiting implemented.

## Caching

Redis caching enabled with 2-hour TTL for analysis results.

## WebSocket Support

Not implemented.

## SDK Support

Python SDK example:

```python
import requests

# Analyze EEG
response = requests.post(
    "http://localhost:8000/api/v1/eeg/analyze",
    json={
        "file_path": "data/sample.edf",
        "analysis_types": ["sleep", "quality"]
    }
)
job_id = response.json()["job_id"]

# Get results
results = requests.get(
    f"http://localhost:8000/api/v1/eeg/results/{job_id}"
)
print(results.json())
```

## OpenAPI Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

All errors return JSON with structure:

```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "File must be EDF or BDF format",
    "details": {...}
  }
}
```

## Performance

- Average response time: <100ms (cached), <30s (processing)
- Max file size: 2GB
- Concurrent requests: 50
- Memory limit per request: 4GB

## CORS

CORS enabled for all origins in development. Production will restrict to specific domains.

## Monitoring

Health metrics exposed at `/metrics` (Prometheus format) - not yet implemented.
