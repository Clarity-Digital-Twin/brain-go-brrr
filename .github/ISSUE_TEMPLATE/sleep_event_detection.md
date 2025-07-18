---
name: Sleep Event Detection (Spindles & Slow Waves)
about: Implement detection algorithms for sleep spindles and slow waves
title: 'feat: Add sleep spindle and slow wave detection to sleep analysis'
labels: enhancement, sleep-analysis
assignees: ''
---

## Problem Statement
Current sleep analysis only provides sleep stage classification. We need to detect specific sleep events (spindles, slow waves) for comprehensive sleep metrics.

## Requirements
1. Implement sleep spindle detection (11-15 Hz, 0.5-2s duration)
2. Implement slow wave detection (0.5-4 Hz, >75μV amplitude)
3. Add K-complex detection for N2 stage validation
4. Calculate event density per sleep stage
5. Generate event-annotated hypnograms

## Technical Approach
```python
# Leverage YASA's event detection algorithms
from yasa import spindles_detect, sw_detect

# Custom implementation for real-time processing
class SleepEventDetector:
    def detect_spindles(self, eeg_data, sf, threshold='auto'):
        # Bandpass filter 11-15 Hz
        # Hilbert transform for envelope
        # Threshold crossing detection
        pass

    def detect_slow_waves(self, eeg_data, sf, amp_threshold=75):
        # Bandpass filter 0.5-4 Hz
        # Peak-to-peak amplitude detection
        # Duration validation
        pass
```

## Reference Implementation
- Use `/reference_repos/yasa/` for algorithm validation
- Follow MNE-Python event structure for compatibility
- Support both automatic and manual threshold modes

## Acceptance Criteria
- [ ] Spindle detection achieves >80% agreement with YASA
- [ ] Slow wave detection matches literature benchmarks
- [ ] Event density calculations per sleep stage
- [ ] Visual validation plots for detected events
- [ ] Integration with existing sleep analysis pipeline
- [ ] Comprehensive test suite with synthetic events

@claude please implement sleep event detection following our TDD approach. Reference the YASA implementation in `/reference_repos/yasa/` but create our own optimized version. Start with tests for spindle detection, then slow waves.
