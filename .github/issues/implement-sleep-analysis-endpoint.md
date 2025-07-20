# Implement Sleep Analysis API Endpoint

## Problem
Sleep analysis service exists but no API endpoint exposes it.

## Requirements
Create `/api/v1/eeg/analyze/sleep` endpoint that:
1. Accepts EDF file upload
2. Runs YASA sleep staging
3. Returns hypnogram and sleep metrics

## Expected Response Format
```json
{
  "hypnogram": {
    "epochs": ["W", "N1", "N2", "N3", "N2", "REM", ...],
    "epoch_duration_sec": 30,
    "total_epochs": 960
  },
  "metrics": {
    "total_sleep_time_min": 420,
    "sleep_efficiency_pct": 87.5,
    "sleep_onset_latency_min": 12,
    "waso_min": 35,
    "rem_latency_min": 90,
    "stages_pct": {
      "wake": 12.5,
      "n1": 5.2,
      "n2": 48.3,
      "n3": 18.7,
      "rem": 15.3
    }
  },
  "quality_indicators": {
    "fragmentation_index": 0.23,
    "arousal_index": 8.5
  },
  "processing_time_sec": 4.2
}
```

## Implementation Steps
1. Create new router in `src/brain_go_brrr/api/routers/sleep.py`
2. Wire up existing `SleepAnalyzer` from `services/sleep_metrics.py`
3. Add response schema with Pydantic
4. Add tests for endpoint
5. Update API documentation

## Acceptance Criteria
- [ ] Endpoint accepts EDF files up to 2GB
- [ ] Returns sleep staging for full night recordings
- [ ] Processing time < 30 seconds for 8-hour recording
- [ ] Includes confidence scores
- [ ] Handles errors gracefully

@clod please work on this autonomously
