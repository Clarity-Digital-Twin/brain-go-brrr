# API Reference

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

⚠️ **Currently Not Implemented** - API is open for development/research use

## Endpoints

### Health Check

#### GET `/api/v1/health`

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-21T10:30:00Z",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is degraded

---

### EEG Analysis

#### POST `/api/v1/eeg/analyze`

Perform full EEG analysis including quality control, feature extraction, and abnormality detection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: EDF/BDF file upload

**Form Data:**
```
edf_file: <binary file data>
```

**Response:**
```json
{
  "flag": "ROUTINE",
  "confidence": 0.85,
  "bad_channels": ["T3", "O2"],
  "quality_metrics": {
    "bad_channel_ratio": 0.1,
    "abnormality_score": 0.3,
    "quality_grade": "GOOD"
  },
  "recommendation": "Standard review recommended",
  "processing_time": 1.5,
  "quality_grade": "GOOD",
  "timestamp": "2025-08-21T10:30:00Z"
}
```

**Triage Flags:**
- `URGENT`: Requires immediate review (high abnormality)
- `EXPEDITE`: Priority review recommended
- `ROUTINE`: Standard review timeline
- `NORMAL`: No abnormalities detected
- `ERROR`: Processing error occurred

**Status Codes:**
- `200 OK`: Analysis completed successfully
- `400 Bad Request`: Invalid file format
- `413 Payload Too Large`: File exceeds 2GB limit
- `500 Internal Server Error`: Processing error

---

### Quality Control Only

#### POST `/api/v1/eeg/quality`

Run quality control analysis only (bad channels, artifacts).

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: EDF/BDF file upload

**Response:**
```json
{
  "bad_channels": ["T3", "O2"],
  "bad_channel_ratio": 0.1,
  "artifacts": {
    "eye_blinks": 15,
    "muscle": 8,
    "heartbeat": 3
  },
  "quality_score": 0.85,
  "quality_grade": "GOOD",
  "processing_time": 0.8
}
```

**Quality Grades:**
- `EXCELLENT`: >95% clean data
- `GOOD`: 85-95% clean data
- `FAIR`: 70-85% clean data
- `POOR`: <70% clean data

---

### Sleep Analysis

#### POST `/api/v1/eeg/sleep`

Perform sleep staging analysis using YASA.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: EDF/BDF file upload

**Response:**
```json
{
  "sleep_stages": {
    "W": 120,    // minutes in Wake
    "N1": 45,    // minutes in N1
    "N2": 210,   // minutes in N2
    "N3": 95,    // minutes in N3
    "REM": 90    // minutes in REM
  },
  "sleep_metrics": {
    "total_sleep_time": 440,  // minutes
    "sleep_efficiency": 0.87,
    "sleep_onset_latency": 12,  // minutes
    "rem_latency": 85,  // minutes
    "waso": 35  // wake after sleep onset in minutes
  },
  "hypnogram": [0, 0, 1, 1, 2, 2, 2, 3, 3, ...],  // 30s epochs
  "confidence": 0.89,
  "processing_time": 2.1
}
```

**Sleep Stage Codes:**
- `0`: Wake
- `1`: N1
- `2`: N2
- `3`: N3
- `4`: REM

---

### Feature Extraction

#### POST `/api/v1/eeg/features`

Extract EEGPT features from EEG data.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: EDF/BDF file upload

**Query Parameters:**
- `window_size`: Window size in seconds (default: 4.0)
- `overlap`: Overlap between windows (0.0-1.0, default: 0.0)

**Response:**
```json
{
  "features": {
    "shape": [150, 2048],  // [n_windows, feature_dim]
    "dtype": "float32"
  },
  "metadata": {
    "n_windows": 150,
    "window_size": 4.0,
    "sampling_rate": 256,
    "n_channels": 20
  },
  "processing_time": 3.2
}
```

---

## Error Responses

All endpoints may return these error formats:

### Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "edf_file"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### Application Error
```json
{
  "detail": "Error message describing the issue",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2025-08-21T10:30:00Z"
}
```

## Rate Limiting

Currently not implemented. In production:
- Default: 100 requests per minute per IP
- File uploads: 10 per minute per IP

## File Requirements

### Supported Formats
- European Data Format (`.edf`)
- BioSemi Data Format (`.bdf`)

### File Limits
- Maximum size: 2GB
- Minimum duration: 30 seconds
- Maximum duration: 24 hours

### Channel Requirements
- Minimum channels: 1
- Maximum channels: 256
- Preferred montage: 10-20 system

## Response Headers

All responses include:

```
X-Request-ID: <unique-request-id>
X-Processing-Time: <time-in-seconds>
Content-Type: application/json
```

## Caching

Redis caching is implemented with:
- Default TTL: 1 hour
- Cache key: SHA256 hash of file content
- Bypass: Add `X-Cache-Control: no-cache` header

## WebSocket Streaming (Planned)

Future endpoint for real-time analysis:
```
ws://localhost:8000/api/v1/eeg/stream
```

## SDK Examples

### Python
```python
import requests

# Upload and analyze EEG file
with open("recording.edf", "rb") as f:
    files = {"edf_file": ("recording.edf", f, "application/octet-stream")}
    response = requests.post(
        "http://localhost:8000/api/v1/eeg/analyze",
        files=files
    )

result = response.json()
print(f"Triage: {result['flag']}")
print(f"Confidence: {result['confidence']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/api/v1/eeg/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "edf_file=@recording.edf"
```

### JavaScript/TypeScript
```javascript
const formData = new FormData();
formData.append('edf_file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/api/v1/eeg/analyze', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Triage: ${result.flag}`);
```

## Interactive API Documentation

When running locally, interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

*Note: This API is for research purposes only. Not intended for clinical use.*
