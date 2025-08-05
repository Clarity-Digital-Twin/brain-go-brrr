# Hierarchical Pipeline Design

## Overview

This document outlines the complete hierarchical architecture for EEG analysis, integrating EEGPT-based abnormality detection, event classification, sleep analysis, and quality control into a unified system.

## System Architecture

```
┌─────────────────┐
│   EEG Input     │
│  (.edf/.bdf)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quality Control │ ◄─── AutoReject
│  (Parallel)     │     (Clean artifacts)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌─────────┐ ┌─────────┐
│ Sleep   │ │Abnormal │ ◄─── EEGPT Features
│Analysis │ │Detection│     (Binary classification)
│(YASA)   │ └────┬────┘
└─────────┘      │
                 ▼
         ┌───────────────┐
         │   Abnormal?   │
         └───────┬───────┘
                 │
         ┌───────┴───────┐
         │               │
         ▼               ▼
    ┌─────────┐    ┌─────────┐
    │ Normal  │    │  Event  │ ◄─── InceptionTime/MiniRocket
    │ (Done)  │    │Detection│     (6-class classification)
    └─────────┘    └─────────┘
```

## Implementation Architecture

### 1. Main Pipeline Controller
```python
class HierarchicalEEGPipeline:
    """
    Unified pipeline orchestrating all analysis components
    """
    
    def __init__(self, config_path="configs/pipeline_config.yaml"):
        self.config = load_config(config_path)
        
        # Initialize all components
        self.quality_controller = QualityController(
            use_autoreject=self.config.quality.use_autoreject,
            ar_params=self.config.quality.autoreject_params
        )
        
        self.sleep_analyzer = SleepAnalyzer(
            use_yasa=self.config.sleep.use_yasa,
            enhance_with_eegpt=self.config.sleep.use_eegpt_features
        )
        
        self.abnormal_detector = AbnormalityDetector(
            checkpoint=self.config.abnormal.eegpt_checkpoint,
            threshold=self.config.abnormal.decision_threshold
        )
        
        self.event_classifier = EventClassifier(
            method=self.config.events.method,  # 'minirocket' or 'inceptiontime'
            checkpoint=self.config.events.checkpoint
        )
        
        # Pipeline settings
        self.parallel_processing = self.config.pipeline.parallel
        self.save_intermediate = self.config.pipeline.save_intermediate
        
    def process_recording(self, eeg_file_path):
        """
        Process complete EEG recording through all stages
        """
        # Load raw data
        raw = self.load_eeg(eeg_file_path)
        
        # Stage 1: Quality Control (always first)
        quality_report, cleaned_raw = self.quality_controller.process(raw)
        
        # Stage 2: Parallel processing of sleep and abnormality
        if self.parallel_processing:
            results = self._parallel_analysis(cleaned_raw)
        else:
            results = self._sequential_analysis(cleaned_raw)
        
        # Stage 3: Conditional event detection
        if results['abnormal']['is_abnormal']:
            results['events'] = self.event_classifier.classify(
                cleaned_raw,
                abnormal_segments=results['abnormal']['segments']
            )
        else:
            results['events'] = None
        
        # Compile final report
        final_report = self.compile_report(
            quality_report, 
            results,
            eeg_file_path
        )
        
        return final_report
    
    def _parallel_analysis(self, cleaned_raw):
        """Run sleep and abnormal detection in parallel"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            sleep_future = executor.submit(
                self.sleep_analyzer.analyze, cleaned_raw
            )
            abnormal_future = executor.submit(
                self.abnormal_detector.detect, cleaned_raw
            )
            
            results = {
                'sleep': sleep_future.result(),
                'abnormal': abnormal_future.result()
            }
        
        return results
```

### 2. Quality Control Stage
```python
class QualityController:
    """
    First stage: Clean and assess data quality
    """
    
    def __init__(self, use_autoreject=True, ar_params=None):
        self.use_autoreject = use_autoreject
        
        if use_autoreject:
            self.autoreject = AutoReject(**ar_params)
        else:
            self.autoreject = None
    
    def process(self, raw):
        """
        Clean data and generate quality report
        """
        # Create epochs for processing
        epochs = create_fixed_length_epochs(raw, duration=8.0)
        
        # Apply AutoReject if enabled
        if self.use_autoreject:
            epochs_clean, reject_log = self.autoreject.fit_transform(epochs)
            quality_score = self._compute_quality_score(reject_log)
        else:
            epochs_clean = epochs
            reject_log = None
            quality_score = 1.0
        
        # Convert back to continuous
        cleaned_raw = epochs_to_continuous(epochs_clean, raw.info)
        
        # Generate report
        quality_report = {
            'quality_score': quality_score,
            'n_bad_channels': len(self._find_bad_channels(raw)),
            'bad_channels': self._find_bad_channels(raw),
            'artifact_summary': self._summarize_artifacts(reject_log),
            'preprocessing_applied': {
                'autoreject': self.use_autoreject,
                'filtering': '0.5-50Hz bandpass',
                'resampling': 256
            }
        }
        
        return quality_report, cleaned_raw
```

### 3. Abnormality Detection Stage
```python
class AbnormalityDetector:
    """
    Second stage: Binary classification (normal/abnormal)
    """
    
    def __init__(self, checkpoint, threshold=0.5):
        # Load EEGPT model
        self.model = self._load_model(checkpoint)
        self.threshold = threshold
        
    def detect(self, cleaned_raw):
        """
        Detect abnormal segments in recording
        """
        # Extract windows
        windows = self._extract_windows(cleaned_raw)
        
        # Get predictions for each window
        predictions = []
        segments = []
        
        for i, window in enumerate(windows):
            # EEGPT feature extraction + classification
            prob_abnormal = self._predict_window(window)
            
            if prob_abnormal > self.threshold:
                segments.append({
                    'start': i * 4.0,  # 50% overlap
                    'end': (i * 4.0) + 8.0,
                    'confidence': float(prob_abnormal),
                    'window_idx': i
                })
            
            predictions.append(prob_abnormal)
        
        # Aggregate results
        is_abnormal = len(segments) > 0
        overall_confidence = np.max(predictions) if predictions else 0.0
        
        return {
            'is_abnormal': is_abnormal,
            'confidence': overall_confidence,
            'segments': segments,
            'window_predictions': predictions,
            'abnormality_percentage': len(segments) / len(windows) * 100
        }
```

### 4. Event Classification Stage
```python
class EventClassifier:
    """
    Third stage: Multi-class event detection (only for abnormal)
    """
    
    def __init__(self, method='minirocket', checkpoint=None):
        self.method = method
        
        if method == 'minirocket':
            self.model = MiniRocketClassifier(
                n_kernels=10000,
                n_classes=6
            )
        elif method == 'inceptiontime':
            self.model = InceptionTimeClassifier(
                n_classes=6,
                depth=6
            )
        
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))
    
    def classify(self, cleaned_raw, abnormal_segments):
        """
        Classify events in abnormal segments only
        """
        event_results = []
        
        for segment in abnormal_segments:
            # Extract segment data
            start_idx = int(segment['start'] * cleaned_raw.info['sfreq'])
            end_idx = int(segment['end'] * cleaned_raw.info['sfreq'])
            segment_data = cleaned_raw.get_data()[:, start_idx:end_idx]
            
            # Classify event type
            event_probs = self.model.predict_proba(segment_data)
            event_type = self.EVENT_CLASSES[np.argmax(event_probs)]
            
            # Find involved channels
            channel_importance = self._compute_channel_importance(
                segment_data, event_type
            )
            
            event_results.append({
                'time_range': [segment['start'], segment['end']],
                'event_type': event_type,
                'confidence': float(np.max(event_probs)),
                'probabilities': {
                    cls: float(p) for cls, p in 
                    zip(self.EVENT_CLASSES, event_probs)
                },
                'involved_channels': channel_importance.argsort()[-5:].tolist(),
                'clinical_significance': self._assess_significance(event_type)
            })
        
        return {
            'n_events': len(event_results),
            'events': event_results,
            'event_summary': self._summarize_events(event_results)
        }
    
    EVENT_CLASSES = ['SPSW', 'GPED', 'PLED', 'EYEM', 'ARTF', 'BCKG']
```

### 5. Sleep Analysis Stage
```python
class SleepAnalyzer:
    """
    Parallel stage: Sleep staging (runs on all recordings)
    """
    
    def __init__(self, use_yasa=True, enhance_with_eegpt=False):
        self.use_yasa = use_yasa
        self.enhance_with_eegpt = enhance_with_eegpt
        
        if enhance_with_eegpt:
            self.eegpt = load_eegpt_for_sleep()
    
    def analyze(self, cleaned_raw):
        """
        Perform complete sleep analysis
        """
        # Check if recording is suitable for sleep analysis
        if not self._is_sleep_recording(cleaned_raw):
            return {'sleep_analysis': 'Not applicable - recording too short'}
        
        # Run YASA
        hypnogram = self._run_yasa(cleaned_raw)
        
        # Enhance with EEGPT if enabled
        if self.enhance_with_eegpt:
            hypnogram = self._enhance_staging(hypnogram, cleaned_raw)
        
        # Calculate metrics
        metrics = calculate_sleep_metrics(hypnogram)
        
        # Detect microstructure
        microstructure = self._analyze_microstructure(
            cleaned_raw, hypnogram
        )
        
        return {
            'hypnogram': hypnogram.tolist(),
            'metrics': metrics,
            'microstructure': microstructure,
            'sleep_quality_index': self._compute_quality_index(metrics),
            'clinical_flags': self._check_clinical_flags(metrics)
        }
```

## Configuration Management

### 1. Pipeline Configuration
```yaml
# configs/pipeline_config.yaml
pipeline:
  parallel: true
  save_intermediate: true
  cache_dir: /data/cache/pipeline

quality:
  use_autoreject: true
  autoreject_params:
    n_interpolate: [1, 2, 3, 4]
    consensus: [0.2, 0.3, 0.4]
    cv: 5
    n_jobs: 4

abnormal:
  eegpt_checkpoint: /data/models/eegpt_tuab_best.pth
  decision_threshold: 0.5
  window_size: 8.0
  window_stride: 4.0

events:
  method: minirocket  # or inceptiontime
  checkpoint: /data/models/event_classifier.pth
  confidence_threshold: 0.7

sleep:
  use_yasa: true
  use_eegpt_features: false
  min_duration_hours: 4
```

### 2. Adaptive Processing
```python
class AdaptivePipeline(HierarchicalEEGPipeline):
    """
    Pipeline that adapts based on data characteristics
    """
    
    def process_recording(self, eeg_file_path):
        # Analyze recording characteristics
        characteristics = self._analyze_recording(eeg_file_path)
        
        # Adapt pipeline configuration
        if characteristics['duration_hours'] < 0.5:
            # Short recording - skip sleep analysis
            self.config.sleep.enabled = False
            
        if characteristics['n_channels'] < 10:
            # Limited channels - adjust quality control
            self.config.quality.autoreject_params['n_interpolate'] = [1, 2]
            
        if characteristics['sampling_rate'] < 200:
            # Low sampling rate - adjust event detection
            self.config.events.confidence_threshold = 0.8
        
        # Run adapted pipeline
        return super().process_recording(eeg_file_path)
```

## Performance Optimization

### 1. Caching Strategy
```python
class CachedPipeline:
    """Implement multi-level caching for efficiency"""
    
    def __init__(self, cache_dir="/data/cache"):
        self.cache_levels = {
            'quality': Path(cache_dir) / 'quality',
            'features': Path(cache_dir) / 'features',
            'predictions': Path(cache_dir) / 'predictions'
        }
        
    def get_or_compute(self, stage, key, compute_fn):
        """Generic caching mechanism"""
        cache_file = self.cache_levels[stage] / f"{key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        result = compute_fn()
        
        cache_file.parent.mkdir(exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        return result
```

### 2. Batch Processing
```python
def batch_process_recordings(file_paths, n_workers=4):
    """Process multiple recordings efficiently"""
    
    # Initialize pipeline pool
    pipeline = HierarchicalEEGPipeline()
    
    # Process in parallel
    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(pipeline.process_recording, file_paths)
    
    # Aggregate results
    summary = {
        'total_processed': len(results),
        'abnormal_count': sum(r['abnormal']['is_abnormal'] for r in results),
        'quality_scores': [r['quality']['quality_score'] for r in results],
        'processing_times': [r['metadata']['processing_time'] for r in results]
    }
    
    return results, summary
```

## Clinical Integration

### 1. Report Generation
```python
class ClinicalReportGenerator:
    """Generate comprehensive clinical reports"""
    
    def generate_report(self, pipeline_results):
        report = {
            'header': self._generate_header(pipeline_results),
            'summary': self._generate_summary(pipeline_results),
            'findings': self._generate_findings(pipeline_results),
            'recommendations': self._generate_recommendations(pipeline_results),
            'technical_details': self._generate_technical(pipeline_results)
        }
        
        # Generate visualizations
        figures = self._generate_figures(pipeline_results)
        
        # Compile PDF report
        pdf_path = self._compile_pdf(report, figures)
        
        return pdf_path
    
    def _generate_findings(self, results):
        findings = []
        
        # Quality findings
        if results['quality']['quality_score'] < 0.7:
            findings.append({
                'type': 'quality',
                'severity': 'moderate',
                'description': 'Recording contains significant artifacts',
                'details': results['quality']['artifact_summary']
            })
        
        # Abnormality findings
        if results['abnormal']['is_abnormal']:
            findings.append({
                'type': 'abnormality',
                'severity': 'high',
                'description': 'Abnormal EEG patterns detected',
                'details': {
                    'segments': len(results['abnormal']['segments']),
                    'confidence': results['abnormal']['confidence']
                }
            })
        
        # Event findings
        if results.get('events'):
            for event in results['events']['events']:
                if event['event_type'] in ['SPSW', 'GPED', 'PLED']:
                    findings.append({
                        'type': 'epileptiform',
                        'severity': 'high',
                        'description': f'{event["event_type"]} detected',
                        'time': event['time_range'],
                        'channels': event['involved_channels']
                    })
        
        # Sleep findings
        if results.get('sleep') and 'metrics' in results['sleep']:
            metrics = results['sleep']['metrics']
            if metrics['sleep_efficiency'] < 80:
                findings.append({
                    'type': 'sleep',
                    'severity': 'moderate',
                    'description': 'Poor sleep efficiency',
                    'details': f"{metrics['sleep_efficiency']:.1f}%"
                })
        
        return findings
```

### 2. Real-Time Monitoring
```python
class RealTimePipeline:
    """Streaming analysis for continuous monitoring"""
    
    def __init__(self, buffer_size=30.0):  # 30 second buffer
        self.buffer = CircularBuffer(buffer_size)
        self.pipeline = HierarchicalEEGPipeline()
        self.alert_manager = AlertManager()
        
    def process_stream(self, data_chunk):
        self.buffer.append(data_chunk)
        
        if self.buffer.is_ready():
            # Extract latest window
            window = self.buffer.get_latest_window()
            
            # Quick abnormality check
            quick_result = self.pipeline.abnormal_detector.quick_detect(window)
            
            if quick_result['abnormal_prob'] > 0.8:
                # High confidence abnormality - full analysis
                full_result = self.pipeline.process_window(window)
                
                # Check for critical events
                if self._is_critical(full_result):
                    self.alert_manager.send_alert(full_result)
            
            return quick_result
```

## Validation Framework

### 1. Pipeline Validation
```python
class PipelineValidator:
    """Validate pipeline components and integration"""
    
    def validate_pipeline(self, test_data_dir):
        results = {
            'component_tests': self._test_components(),
            'integration_tests': self._test_integration(),
            'performance_tests': self._test_performance(test_data_dir),
            'clinical_tests': self._test_clinical_accuracy(test_data_dir)
        }
        
        return results
    
    def _test_integration(self):
        """Test component interactions"""
        # Create synthetic test data
        test_raw = create_test_eeg()
        
        # Test quality → abnormal flow
        quality_report, cleaned = QualityController().process(test_raw)
        abnormal_result = AbnormalityDetector().detect(cleaned)
        
        assert cleaned.get_data().shape == test_raw.get_data().shape
        assert 0 <= abnormal_result['confidence'] <= 1
        
        # Test abnormal → event flow
        if abnormal_result['is_abnormal']:
            event_result = EventClassifier().classify(
                cleaned, 
                abnormal_result['segments']
            )
            assert len(event_result['events']) <= len(abnormal_result['segments'])
        
        return "Integration tests passed"
```

### 2. Clinical Validation
```python
def validate_against_expert_annotations(pipeline, annotated_dataset):
    """Compare pipeline results with expert annotations"""
    
    results = []
    
    for recording, annotations in annotated_dataset:
        # Run pipeline
        pipeline_output = pipeline.process_recording(recording)
        
        # Compare with expert
        comparison = {
            'recording': recording,
            'abnormal_agreement': compare_abnormal_detection(
                pipeline_output['abnormal'],
                annotations['abnormal']
            ),
            'event_agreement': compare_event_detection(
                pipeline_output.get('events'),
                annotations.get('events')
            ),
            'sleep_agreement': compare_sleep_staging(
                pipeline_output.get('sleep'),
                annotations.get('sleep')
            )
        }
        
        results.append(comparison)
    
    # Aggregate metrics
    summary = {
        'abnormal_accuracy': np.mean([r['abnormal_agreement']['accuracy'] for r in results]),
        'event_f1': np.mean([r['event_agreement']['f1'] for r in results if r['event_agreement']]),
        'sleep_kappa': np.mean([r['sleep_agreement']['kappa'] for r in results if r['sleep_agreement']])
    }
    
    return results, summary
```

## Deployment Architecture

### 1. API Endpoints
```python
from fastapi import FastAPI, UploadFile, BackgroundTasks

app = FastAPI()

@app.post("/analyze/full")
async def analyze_full(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    """Full hierarchical analysis"""
    # Save uploaded file
    file_path = save_upload(file)
    
    # Queue analysis
    job_id = str(uuid.uuid4())
    background_tasks.add_task(
        run_pipeline_analysis,
        job_id,
        file_path
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "estimated_time": estimate_processing_time(file_path)
    }

@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Retrieve analysis results"""
    results = get_cached_results(job_id)
    
    if results is None:
        return {"status": "processing"}
    
    return {
        "status": "complete",
        "results": results,
        "report_url": f"/reports/{job_id}.pdf"
    }
```

### 2. Scalability Considerations
```python
class ScalablePipeline:
    """Pipeline designed for horizontal scaling"""
    
    def __init__(self, redis_url, s3_bucket):
        self.task_queue = RedisQueue(redis_url)
        self.storage = S3Storage(s3_bucket)
        self.workers = []
    
    def scale_workers(self, n_workers):
        """Dynamically scale processing workers"""
        current = len(self.workers)
        
        if n_workers > current:
            # Scale up
            for _ in range(n_workers - current):
                worker = PipelineWorker(self.task_queue, self.storage)
                worker.start()
                self.workers.append(worker)
        else:
            # Scale down
            for _ in range(current - n_workers):
                worker = self.workers.pop()
                worker.stop()
```

## Future Enhancements

1. **Multi-Modal Integration**: Combine EEG with video/audio
2. **Continuous Learning**: Update models with clinician feedback
3. **Explainable AI**: Generate attention maps and feature importance
4. **Cross-Dataset Training**: Improve generalization
5. **Edge Deployment**: Run on local devices for privacy

## References

- Architecture inspired by clinical workflow
- EEGPT: Feature extraction backbone
- AutoReject: Quality control
- YASA: Sleep analysis
- InceptionTime/MiniRocket: Event detection
- Integration: brain_go_brrr/pipelines/hierarchical.py