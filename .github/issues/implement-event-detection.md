# Implement Event Detection Service

## Problem
Event detection for epileptiform discharges is specified in requirements but not implemented.

## Requirements
Implement service to detect:
1. Epileptiform discharges (spikes, sharp waves)
2. GPED/PLED patterns
3. Seizure activity
4. Other clinically relevant events

## Technical Approach
1. Use EEGPT features as input
2. Implement sliding window detection
3. Add temporal clustering to reduce false positives
4. Include confidence scoring

## API Endpoint
`/api/v1/eeg/analyze/events`

## Expected Response
```json
{
  "events": [
    {
      "type": "spike",
      "start_time_sec": 125.3,
      "duration_ms": 70,
      "channels": ["F3", "F4"],
      "confidence": 0.85,
      "morphology": {
        "amplitude_uv": 120,
        "sharpness": 0.9
      }
    },
    {
      "type": "seizure",
      "start_time_sec": 1823.0,
      "duration_sec": 45,
      "channels": ["all"],
      "confidence": 0.92,
      "pattern": "generalized_tonic_clonic"
    }
  ],
  "summary": {
    "total_events": 15,
    "events_per_hour": 1.2,
    "seizure_burden_pct": 0.5
  }
}
```

## Implementation Steps
1. Research existing spike detection algorithms
2. Create `EventDetector` class in `src/brain_go_brrr/core/events/`
3. Integrate with EEGPT feature extraction
4. Add comprehensive tests with synthetic events
5. Validate against clinical annotations

## Acceptance Criteria
- [ ] Detects spikes with >80% sensitivity
- [ ] False positive rate <20%
- [ ] Processes 1 hour in <10 seconds
- [ ] Includes detailed event metadata
- [ ] Handles multichannel patterns

@clod please work on this autonomously
