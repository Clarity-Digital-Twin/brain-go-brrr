# Sleep Analysis Integration

## Overview

This document details the integration of YASA (Yet Another Spindle Algorithm) for automated sleep staging in the Brain-Go-Brrr system. YASA achieves state-of-the-art performance with 87.46% accuracy and runs independently of the abnormality detection pipeline.

## Key Performance Metrics

### YASA Benchmarks
- **Overall Accuracy**: 87.46% (median across 585 test nights)
- **Cohen's Kappa**: 0.819 (excellent agreement)
- **N3 Detection**: 83.2% sensitivity, F1=0.835
- **REM Detection**: >85% sensitivity, F1≥0.86
- **Processing**: <2 minutes for 8-hour recording

### Stage-Specific Performance
| Sleep Stage | Sensitivity | F1-Score | Common Confusion |
|-------------|------------|----------|------------------|
| Wake | >85% | ≥0.86 | → N1 (transitions) |
| N1 | 45.4% | 0.432 | → N2 (27.5%) |
| N2 | >85% | ≥0.86 | → N3/N1 |
| N3 | 83.2% | 0.835 | → N2 (16.1%) |
| REM | >85% | ≥0.86 | Rarely confused |

## Architecture Design

### 1. Independent Sleep Analysis Pipeline
```python
class SleepAnalysisPipeline:
    """
    Runs independently of abnormality detection
    Processes ALL recordings (normal and abnormal)
    """
    
    def __init__(self):
        # YASA configuration
        self.yasa_config = {
            'eeg_name': 'C3-M2',  # Primary channel
            'eog_name': 'E1-M2',  # Eye movement
            'emg_name': 'EMG1-EMG2',  # Muscle activity
            'sf': 256,  # Sampling frequency
        }
        
        # EEGPT for enhanced features (optional)
        self.use_eegpt_features = True
        self.eegpt = load_eegpt() if self.use_eegpt_features else None
    
    def analyze(self, eeg_file):
        # Load PSG data
        raw = mne.io.read_raw_edf(eeg_file)
        
        # Preprocess for sleep staging
        data = self.preprocess_for_sleep(raw)
        
        # Run YASA
        hypnogram = self.run_yasa(data)
        
        # Calculate metrics
        metrics = self.calculate_sleep_metrics(hypnogram)
        
        # Generate report
        report = self.generate_sleep_report(hypnogram, metrics)
        
        return report
```

### 2. Data Preprocessing for Sleep
```python
def preprocess_for_sleep(raw, target_sfreq=256):
    """
    Specific preprocessing for sleep staging
    Different from abnormality detection preprocessing
    """
    # Select required channels
    channels = {
        'eeg': ['C3-M2', 'C4-M1'],  # Central EEG
        'eog': ['E1-M2', 'E2-M1'],  # Eye movements
        'emg': ['EMG1-EMG2']         # Chin EMG
    }
    
    # Create bipolar montage if needed
    if 'C3-M2' not in raw.ch_names:
        raw = create_bipolar_montage(raw)
    
    # Resample to standard frequency
    raw.resample(target_sfreq)
    
    # Filter specifications for sleep
    raw.filter(l_freq=0.3, h_freq=35.0, picks='eeg')
    raw.filter(l_freq=0.3, h_freq=35.0, picks='eog')
    raw.filter(l_freq=10.0, h_freq=100.0, picks='emg')
    
    # Artifact suppression (gentler than AutoReject)
    raw = suppress_artifacts_for_sleep(raw)
    
    return raw
```

### 3. YASA Integration
```python
import yasa

class YASASleepStager:
    def __init__(self, eegpt_features=None):
        self.eegpt_features = eegpt_features
        
    def stage_sleep(self, raw_psg):
        """
        Run YASA sleep staging with optional EEGPT enhancement
        """
        # Extract channels
        eeg = raw_psg.get_data(picks=['C3-M2'])[0]
        eog = raw_psg.get_data(picks=['E1-M2'])[0]
        emg = raw_psg.get_data(picks=['EMG1-EMG2'])[0]
        sf = raw_psg.info['sfreq']
        
        # Standard YASA staging
        sls = yasa.SleepStaging(
            eeg=eeg, 
            eog=eog, 
            emg=emg, 
            sf=sf
        )
        
        # Get predictions
        hypno_30s = sls.predict()  # 30-second epochs
        
        # Optional: Enhance with EEGPT features
        if self.eegpt_features is not None:
            hypno_30s = self.enhance_with_eegpt(
                hypno_30s, 
                self.eegpt_features,
                sls.proba  # YASA probabilities
            )
        
        # Convert to different resolutions
        hypno_5s = yasa.hypno_upsample_to_data(
            hypno_30s, 
            sf_hypno=1/30, 
            data=eeg, 
            sf_data=sf
        )
        
        return {
            'hypnogram_30s': hypno_30s,
            'hypnogram_5s': hypno_5s,
            'probabilities': sls.proba,
            'confidence': sls.proba.max(axis=1).mean()
        }
```

### 4. Sleep Metrics Calculation
```python
def calculate_sleep_metrics(hypnogram, sf_hypno=1/30):
    """
    Calculate comprehensive sleep metrics
    """
    # Basic metrics
    metrics = {}
    
    # Total recording time
    trt_min = len(hypnogram) * 30 / 60  # minutes
    
    # Sleep period time (first sleep to last sleep)
    sleep_indices = np.where(hypnogram != 0)[0]
    if len(sleep_indices) > 0:
        spt_min = (sleep_indices[-1] - sleep_indices[0] + 1) * 30 / 60
    else:
        spt_min = 0
    
    # Total sleep time
    tst_min = np.sum(hypnogram != 0) * 30 / 60
    
    # Sleep efficiency
    sleep_efficiency = (tst_min / trt_min) * 100 if trt_min > 0 else 0
    
    # Stage percentages
    stage_counts = pd.Series(hypnogram).value_counts()
    stage_pct = {}
    for stage in [0, 1, 2, 3, 4]:  # W, N1, N2, N3, REM
        count = stage_counts.get(stage, 0)
        stage_pct[f'stage_{stage}_pct'] = (count / len(hypnogram)) * 100
    
    # Sleep latencies
    sleep_latency = calculate_sleep_latency(hypnogram)
    rem_latency = calculate_rem_latency(hypnogram)
    
    # WASO (Wake After Sleep Onset)
    waso = calculate_waso(hypnogram)
    
    # Number of awakenings
    n_awakenings = count_awakenings(hypnogram)
    
    # Compile metrics
    metrics.update({
        'trt_min': trt_min,
        'spt_min': spt_min,
        'tst_min': tst_min,
        'sleep_efficiency': sleep_efficiency,
        'sleep_latency_min': sleep_latency,
        'rem_latency_min': rem_latency,
        'waso_min': waso,
        'n_awakenings': n_awakenings,
        'n1_pct': stage_pct.get('stage_1_pct', 0),
        'n2_pct': stage_pct.get('stage_2_pct', 0),
        'n3_pct': stage_pct.get('stage_3_pct', 0),
        'rem_pct': stage_pct.get('stage_4_pct', 0),
        'wake_pct': stage_pct.get('stage_0_pct', 0)
    })
    
    return metrics
```

### 5. Advanced Sleep Features
```python
class AdvancedSleepAnalysis:
    """Extended sleep analysis beyond basic staging"""
    
    def analyze_sleep_architecture(self, hypnogram):
        """Analyze sleep cycles and transitions"""
        # Detect sleep cycles
        cycles = self.detect_sleep_cycles(hypnogram)
        
        # Transition matrix
        transitions = self.calculate_transitions(hypnogram)
        
        # Stage stability
        stability = self.calculate_stage_stability(hypnogram)
        
        return {
            'n_cycles': len(cycles),
            'cycle_durations': [c['duration'] for c in cycles],
            'transition_matrix': transitions,
            'stage_stability': stability
        }
    
    def detect_sleep_cycles(self, hypnogram):
        """Identify NREM-REM cycles"""
        cycles = []
        current_cycle = None
        
        for i, stage in enumerate(hypnogram):
            if stage in [1, 2, 3]:  # NREM stages
                if current_cycle is None:
                    current_cycle = {'start': i, 'has_rem': False}
            elif stage == 4:  # REM
                if current_cycle is not None:
                    current_cycle['has_rem'] = True
            elif stage == 0:  # Wake
                if current_cycle and current_cycle['has_rem']:
                    current_cycle['end'] = i
                    current_cycle['duration'] = (i - current_cycle['start']) * 30 / 60
                    cycles.append(current_cycle)
                    current_cycle = None
        
        return cycles
    
    def detect_microevents(self, raw_psg, hypnogram):
        """Detect spindles, slow waves, and other events"""
        results = {}
        
        # Sleep spindles (during N2)
        n2_mask = hypnogram == 2
        if np.any(n2_mask):
            sp = yasa.spindles_detect(
                raw_psg.get_data(picks=['C3-M2'])[0],
                raw_psg.info['sfreq'],
                hypno=hypnogram
            )
            results['spindles'] = sp.summary() if sp else None
        
        # Slow waves (during N3)
        n3_mask = hypnogram == 3
        if np.any(n3_mask):
            sw = yasa.sw_detect(
                raw_psg.get_data(picks=['C3-M2'])[0],
                raw_psg.info['sfreq'],
                hypno=hypnogram
            )
            results['slow_waves'] = sw.summary() if sw else None
        
        # REM density
        rem_mask = hypnogram == 4
        if np.any(rem_mask):
            results['rem_density'] = self.calculate_rem_density(
                raw_psg, hypnogram
            )
        
        return results
```

### 6. Integration with Abnormality Detection
```python
class IntegratedAnalysis:
    """
    Combine sleep staging with abnormality detection
    Sleep-state dependent abnormality analysis
    """
    
    def __init__(self):
        self.sleep_analyzer = SleepAnalysisPipeline()
        self.abnormal_detector = AbnormalityDetectionPipeline()
    
    def analyze_recording(self, eeg_file):
        # Run both pipelines in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            sleep_future = executor.submit(
                self.sleep_analyzer.analyze, eeg_file
            )
            abnormal_future = executor.submit(
                self.abnormal_detector.analyze, eeg_file
            )
            
            sleep_results = sleep_future.result()
            abnormal_results = abnormal_future.result()
        
        # Combine results
        integrated_results = self.integrate_findings(
            sleep_results, 
            abnormal_results
        )
        
        return integrated_results
    
    def integrate_findings(self, sleep_results, abnormal_results):
        """Correlate abnormalities with sleep stages"""
        hypnogram = sleep_results['hypnogram_30s']
        abnormal_windows = abnormal_results['abnormal_segments']
        
        # Map abnormalities to sleep stages
        stage_abnormalities = {
            'wake': [],
            'n1': [],
            'n2': [],
            'n3': [],
            'rem': []
        }
        
        stage_map = {0: 'wake', 1: 'n1', 2: 'n2', 3: 'n3', 4: 'rem'}
        
        for segment in abnormal_windows:
            epoch_idx = int(segment['start'] / 30)
            if epoch_idx < len(hypnogram):
                stage = stage_map[hypnogram[epoch_idx]]
                stage_abnormalities[stage].append(segment)
        
        # Calculate stage-specific abnormality rates
        abnormality_rates = {}
        for stage, abnorms in stage_abnormalities.items():
            stage_epochs = np.sum(hypnogram == list(stage_map.keys())[
                list(stage_map.values()).index(stage)
            ])
            if stage_epochs > 0:
                abnormality_rates[stage] = len(abnorms) / stage_epochs
            else:
                abnormality_rates[stage] = 0
        
        return {
            'sleep_metrics': sleep_results['metrics'],
            'abnormality_summary': abnormal_results['summary'],
            'stage_specific_abnormalities': stage_abnormalities,
            'abnormality_rates_by_stage': abnormality_rates,
            'clinical_correlations': self.identify_clinical_patterns(
                stage_abnormalities, sleep_results
            )
        }
```

## Data Requirements

### 1. Channel Requirements
```python
REQUIRED_CHANNELS = {
    'eeg': ['C3', 'C4', 'M1', 'M2'],  # Central EEG + mastoids
    'eog': ['E1', 'E2'],                # Eye movements
    'emg': ['EMG1', 'EMG2', 'EMG3'],   # Chin EMG
    'optional': ['F3', 'F4', 'O1', 'O2'] # Additional EEG
}

# Fallback options
CHANNEL_ALTERNATIVES = {
    'C3-M2': ['C3-A2', 'C3-REF'],
    'C4-M1': ['C4-A1', 'C4-REF'],
    'E1-M2': ['LOC-A2', 'LOC-REF'],
    'E2-M1': ['ROC-A1', 'ROC-REF']
}
```

### 2. Recording Requirements
- **Duration**: Minimum 4 hours (preferably full night)
- **Sampling Rate**: 100-500 Hz (resampled to 256 Hz)
- **Filters**: Minimal filtering to preserve sleep features

## Clinical Output Format

### 1. Sleep Report Structure
```json
{
    "recording_id": "PSG_2024_001",
    "recording_duration_hours": 8.2,
    "analysis_timestamp": "2024-01-20T08:30:00Z",
    
    "sleep_summary": {
        "total_sleep_time_min": 432,
        "sleep_efficiency_pct": 88.5,
        "sleep_latency_min": 12,
        "rem_latency_min": 95,
        "waso_min": 45,
        "n_awakenings": 8
    },
    
    "stage_distribution": {
        "wake_pct": 11.5,
        "n1_pct": 5.2,
        "n2_pct": 48.3,
        "n3_pct": 18.7,
        "rem_pct": 16.3
    },
    
    "sleep_architecture": {
        "n_complete_cycles": 5,
        "mean_cycle_duration_min": 92,
        "sleep_fragmentation_index": 12.3
    },
    
    "microstructure": {
        "spindle_density": 3.2,
        "slow_wave_density": 45.6,
        "rem_density": 68.2
    },
    
    "clinical_flags": {
        "reduced_rem": false,
        "reduced_n3": false,
        "high_fragmentation": false,
        "abnormal_latencies": false
    },
    
    "visualization": {
        "hypnogram_url": "/api/v1/sleep/hypnogram/PSG_2024_001.png",
        "spectrogram_url": "/api/v1/sleep/spectrogram/PSG_2024_001.png"
    }
}
```

### 2. Hypnogram Visualization
```python
def create_hypnogram_plot(hypnogram, metrics):
    """Generate clinical hypnogram visualization"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Main hypnogram
    time_hours = np.arange(len(hypnogram)) * 30 / 3600
    
    # Invert stages for traditional display
    display_hypno = 4 - hypnogram  # Wake at top
    display_hypno[hypnogram == 4] = 0  # REM at bottom
    
    ax1.step(time_hours, display_hypno, where='post', color='navy')
    ax1.fill_between(time_hours, display_hypno, 4, step='post', alpha=0.3)
    
    # Labels
    ax1.set_yticks([0, 1, 2, 3, 4])
    ax1.set_yticklabels(['REM', 'N3', 'N2', 'N1', 'Wake'])
    ax1.set_xlabel('Time (hours)')
    ax1.set_title('Sleep Hypnogram')
    ax1.grid(True, alpha=0.3)
    
    # Sleep cycles
    cycles = detect_sleep_cycles(hypnogram)
    for i, cycle in enumerate(cycles):
        start_h = cycle['start'] * 30 / 3600
        end_h = cycle['end'] * 30 / 3600
        ax1.axvspan(start_h, end_h, alpha=0.1, color='green')
        ax1.text(start_h + (end_h - start_h)/2, 4.5, f'C{i+1}', 
                ha='center', fontsize=8)
    
    # Stage percentages bar
    stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
    percentages = [
        metrics['wake_pct'],
        metrics['n1_pct'],
        metrics['n2_pct'],
        metrics['n3_pct'],
        metrics['rem_pct']
    ]
    colors = ['lightcoral', 'lightyellow', 'lightblue', 'darkblue', 'purple']
    
    ax2.barh(0, percentages, left=np.cumsum([0] + percentages[:-1]),
             color=colors, height=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Stage Distribution (%)')
    ax2.set_yticks([])
    
    # Add percentage labels
    cumsum = 0
    for i, (stage, pct) in enumerate(zip(stages, percentages)):
        if pct > 5:  # Only label if >5%
            ax2.text(cumsum + pct/2, 0, f'{stage}\n{pct:.1f}%',
                    ha='center', va='center', fontsize=9)
        cumsum += pct
    
    plt.tight_layout()
    return fig
```

## Performance Optimization

### 1. Parallel Processing
```python
def batch_sleep_analysis(eeg_files, n_workers=4):
    """Process multiple recordings in parallel"""
    
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(analyze_single_recording, eeg_files)
    
    return results

def analyze_single_recording(eeg_file):
    """Wrapper for multiprocessing"""
    try:
        analyzer = SleepAnalysisPipeline()
        return analyzer.analyze(eeg_file)
    except Exception as e:
        logger.error(f"Failed to analyze {eeg_file}: {e}")
        return None
```

### 2. Caching Strategy
```python
# Cache preprocessed PSG data
@lru_cache(maxsize=100)
def load_and_preprocess_psg(file_path, cache_dir="/tmp/sleep_cache"):
    cache_file = Path(cache_dir) / f"{hash(file_path)}.pkl"
    
    if cache_file.exists():
        return pickle.load(open(cache_file, 'rb'))
    
    raw = mne.io.read_raw_edf(file_path)
    processed = preprocess_for_sleep(raw)
    
    pickle.dump(processed, open(cache_file, 'wb'))
    return processed
```

## Quality Control

### 1. Data Quality Checks
```python
def check_sleep_data_quality(raw_psg):
    """Ensure data quality for reliable sleep staging"""
    
    quality_report = {
        'has_required_channels': True,
        'adequate_duration': True,
        'signal_quality': {},
        'warnings': []
    }
    
    # Check channels
    required = ['C3-M2', 'E1-M2', 'EMG1-EMG2']
    missing = [ch for ch in required if ch not in raw_psg.ch_names]
    if missing:
        quality_report['has_required_channels'] = False
        quality_report['warnings'].append(f"Missing channels: {missing}")
    
    # Check duration
    duration_hours = raw_psg.times[-1] / 3600
    if duration_hours < 4:
        quality_report['adequate_duration'] = False
        quality_report['warnings'].append(
            f"Recording too short: {duration_hours:.1f} hours"
        )
    
    # Check signal quality
    for ch_type in ['eeg', 'eog', 'emg']:
        snr = calculate_snr(raw_psg, ch_type)
        quality_report['signal_quality'][ch_type] = {
            'snr': snr,
            'acceptable': snr > 10  # dB
        }
    
    return quality_report
```

### 2. Result Validation
```python
def validate_sleep_results(hypnogram, metrics):
    """Sanity checks on sleep analysis results"""
    
    issues = []
    
    # Check for impossible values
    if metrics['sleep_efficiency'] > 100:
        issues.append("Sleep efficiency > 100%")
    
    # Check for unusual patterns
    if metrics['n3_pct'] < 5 and metrics['age'] < 30:
        issues.append("Unusually low N3 for young adult")
    
    if metrics['rem_pct'] < 10:
        issues.append("Unusually low REM percentage")
    
    # Check hypnogram consistency
    if len(np.unique(hypnogram)) < 3:
        issues.append("Less than 3 sleep stages detected")
    
    # Check for extended wake periods
    wake_bouts = find_continuous_segments(hypnogram, stage=0)
    long_wakes = [b for b in wake_bouts if b['duration'] > 60]
    if len(long_wakes) > 3:
        issues.append(f"{len(long_wakes)} wake periods >60 min")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'confidence': 1.0 - (len(issues) * 0.1)
    }
```

## Clinical Applications

### 1. Sleep Disorder Detection
```python
def screen_for_sleep_disorders(sleep_results):
    """Flag potential sleep disorders based on metrics"""
    
    flags = {}
    metrics = sleep_results['metrics']
    
    # Insomnia indicators
    if (metrics['sleep_latency_min'] > 30 or 
        metrics['waso_min'] > 30 or
        metrics['sleep_efficiency'] < 85):
        flags['insomnia_risk'] = 'HIGH'
    
    # REM behavior disorder risk
    if metrics['rem_pct'] < 15 and metrics['age'] > 50:
        flags['rbd_screen'] = 'RECOMMENDED'
    
    # Circadian rhythm disorder
    sleep_midpoint = calculate_sleep_midpoint(sleep_results['hypnogram'])
    if sleep_midpoint < 2 or sleep_midpoint > 6:
        flags['circadian_disorder'] = 'POSSIBLE'
    
    return flags
```

### 2. Treatment Monitoring
```python
def compare_sleep_studies(baseline, followup):
    """Track changes in sleep architecture over time"""
    
    changes = {}
    
    for metric in ['tst_min', 'sleep_efficiency', 'n3_pct', 'rem_pct']:
        baseline_val = baseline['metrics'][metric]
        followup_val = followup['metrics'][metric]
        
        change_pct = ((followup_val - baseline_val) / baseline_val) * 100
        changes[metric] = {
            'baseline': baseline_val,
            'followup': followup_val,
            'change_pct': change_pct,
            'improved': change_pct > 5
        }
    
    return changes
```

## Integration Examples

### Complete Pipeline
```python
# Initialize components
sleep_pipeline = SleepAnalysisPipeline()
abnormal_pipeline = AbnormalityDetectionPipeline()
event_detector = EventDetectionPipeline()

# Process recording
eeg_file = "patient_001_psg.edf"

# Run all analyses
results = {
    'sleep': sleep_pipeline.analyze(eeg_file),
    'abnormalities': abnormal_pipeline.analyze(eeg_file),
    'events': event_detector.analyze(eeg_file)
}

# Generate integrated report
report = generate_comprehensive_report(results)
```

## References

- YASA Paper: "An open-source, high-performance tool for automated sleep staging"
- Implementation: https://github.com/raphaelvallat/yasa
- Sleep Scoring Manual: AASM Manual for Scoring Sleep
- Integration: brain_go_brrr/services/sleep_metrics.py