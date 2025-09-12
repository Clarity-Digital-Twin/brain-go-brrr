# 🛠️ TUSZ Implementation Guide

**Created**: September 9, 2025  
**Status**: 🔥 READY TO CODE  
**Type**: Execution Guide with Code Examples for SeizureTransformer Integration  
**Companion**: See TUSZ_SPEC.md for requirements, TUSZ_ROADMAP.md for execution plan

---

## 📦 Quick Start Commands

```bash
# Setup environment
cd brain-go-brrr
uv sync

# Download TUSZ data (requires agreement)
./scripts/download_tusz.sh

# Get SeizureTransformer weights
wget https://github.com/wu-2025/seizure-transformer/releases/download/v1.0/model.pth \
  -O data/models/pretrained/seizure_transformer_wu2025.pth

# Run first experiment
uv run python experiments/tusz/evaluate_seizure_transformer.py
```

---

## 🏗️ Project Structure

```
brain-go-brrr/
├── src/brain_go_brrr/
│   ├── infra/
│   │   ├── data/
│   │   │   └── tusz_dataset.py          # Dataset loader
│   │   ├── ml_models/
│   │   │   ├── seizure_transformer_wrapper.py   # Model wrapper (no sys.path hacks)
│   │   │   └── temporal_heads.py                # BiLSTM, GRU heads
│   │   └── eval/
│   │       ├── nedc_wrapper.py                  # NEDC adapter surface
│   │       └── post_processing.py               # Hysteresis + merge + min-dur
│   └── domain/
│       └── temporal/
│           └── tusz_controller.py       # Orchestration
├── experiments/
│   └── tusz/
│       ├── configs/
│       │   ├── seizure_transformer.yaml
│       │   └── eegpt_bilstm.yaml
│       ├── evaluate_seizure_transformer.py
│       └── train_eegpt_bilstm.py
└── reference_repos/
    ├── nedc_eeg_eval_v6.0.0/            # NEDC evaluation
    └── SeizureTransformer/              # Wu 2025 model
```

---

## 📊 Data Pipeline Implementation

### Dataset Loader (ensure unipolar montage)
```python
# src/brain_go_brrr/infra/data/tusz_detection_dataset.py

import numpy as np
import pandas as pd
from pathlib import Path
import mne
from typing import Tuple, List, Dict

class TUSZDetectionDataset:
    """
    TUSZ v2.0.1 dataset loader with annotation parsing.
    Handles both EDF files and TSE/CSV annotations.
    """
    
    def __init__(self, root_dir: Path, split: str = 'train', fs: int = 256,
                 window_sec: float = 12.0, stride_sec: float = 1.0,
                 positive_fraction: float = 0.2):
        self.root_dir = Path(root_dir)
        self.split = split
        self.data_dir = self.root_dir / split
        self.fs = fs
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.positive_fraction = positive_fraction
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> Dict[str, List[Tuple[float, float]]]:
        """Load seizure annotations from TSE files."""
        annotations = {}
        
        for tse_file in self.data_dir.glob('**/*.tse'):
            patient_id = tse_file.stem
            events = []
            
            with open(tse_file, 'r') as f:
                for line in f:
                    if 'seiz' in line.lower():
                        parts = line.strip().split()
                        start_sec = float(parts[0])
                        end_sec = float(parts[1])
                        events.append((start_sec, end_sec))
            
            annotations[patient_id] = events
        
        return annotations
    
    def load_recording(self, patient_id: str) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Load EEG recording and annotations for a patient.
        
        Returns:
            eeg_data: (n_channels, n_samples) array
            seizure_events: List of (start_sec, end_sec) tuples
        """
        # Load EDF file
        edf_path = self.data_dir / f"{patient_id}.edf"
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Standardize channels
        raw = self._standardize_channels(raw)
        
        # Get data
        eeg_data = raw.get_data()
        
        # Get annotations
        seizure_events = self.annotations.get(patient_id, [])
        
        return eeg_data, seizure_events
    
    def _standardize_channels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Ensure standard 10-20 montage (19 channels; see SPEC Channel Policy).

        Note: SeizureTransformer requires UNIPOLAR montage. If your recording is bipolar
        (channels with names like "Fp1-F7"), convert to a referential montage first, e.g.:

            raw.set_eeg_reference('average')

        or to a specific reference electrode per your policy. Then resample to 256 Hz.
        """
        standard_channels = [
            'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
            'T3', 'C3', 'CZ', 'C4', 'T4',
            'T5', 'P3', 'PZ', 'P4', 'T6', 'O1', 'O2'
        ]
        
        # Select available channels
        available = [ch for ch in standard_channels if ch in raw.ch_names]
        raw.pick_channels(available, ordered=True)
        
        # Resample to SSOT sampling rate if needed
        if int(raw.info['sfreq']) != self.fs:
            raw.resample(self.fs)
        
        return raw
```

### Annotation Parser
```python
# src/brain_go_brrr/infra/data/tusz_annotations.py (optional utility)

class TUSZAnnotationParser:
    """
    Parse various TUSZ annotation formats (TSE, CSV, XML).
    """
    
    @staticmethod
    def parse_tse(file_path: Path) -> List[Dict]:
        """Parse TSE (Time-Stamped Event) files."""
        events = []
        
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        events.append({
                            'start': float(parts[0]),
                            'end': float(parts[1]),
                            'label': parts[2],
                            'confidence': float(parts[3]) if len(parts) > 3 else 1.0
                        })
        
        return events
    
    @staticmethod
    def events_to_binary_mask(events: List[Dict], duration_sec: float, fs: int = 256) -> np.ndarray:
        """Convert events to binary mask for frame-level evaluation."""
        n_samples = int(duration_sec * fs)
        mask = np.zeros(n_samples, dtype=bool)
        
        for event in events:
            start_sample = int(event['start'] * fs)
            end_sample = int(event['end'] * fs)
            mask[start_sample:end_sample] = True
        
        return mask
```

---

## 🧠 Model Implementations

### SeizureTransformer Wrapper (no sys.path hacks)
```python
# src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py

import torch
import torch.nn as nn
from pathlib import Path

# Preferred: pass a `build_fn` or installed package provides `build_seizure_transformer(n_channels)`
from typing import Optional, Callable

class SeizureTransformerWrapper:
    def __init__(self, weights_path: Optional[Path] = None,
                 build_fn: Optional[Callable[[int], nn.Module]] = None,
                 n_channels: int = 19, fs: int = 256):
        self.fs = fs
        self.window_sec = 60
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if build_fn is None:
            from seizure_transformer import build_seizure_transformer  # installed package
            build_fn = build_seizure_transformer
        self.model = build_fn(n_channels)
        if weights_path is not None and Path(weights_path).exists():
            ckpt = torch.load(weights_path, map_location='cpu', weights_only=True)
            state_dict = ckpt.get('model', ckpt)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
    
    def predict(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Generate per-sample seizure probabilities.
        
        Args:
            eeg_data: (n_channels, n_samples) EEG data
            
        Returns:
            probabilities: (n_samples,) seizure probability per sample
        """
        n_channels, n_samples = eeg_data.shape
        window_samples = self.window_sec * self.fs
        
        # Process in sliding windows
        probabilities = np.zeros(n_samples, dtype=np.float32)
        counts = np.zeros(n_samples, dtype=np.int32)
        stride = window_samples // 2

        for start in range(0, n_samples - window_samples + 1, stride):
            end = start + window_samples
            window = eeg_data[:, start:end]
            
            # Prepare input
            x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                p = torch.sigmoid(self.model(x)).squeeze().detach().cpu().float().mean().item()
            probabilities[start:end] += np.float32(p)
            counts[start:end] += 1

        counts = np.maximum(counts, 1)
        probabilities = probabilities / counts
        return probabilities
```

### EEGPT + BiLSTM Implementation
```python
# src/brain_go_brrr/infra/ml_models/temporal_heads.py

import torch
import torch.nn as nn
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt

class EEGPTBiLSTM(nn.Module):
    """
    EEGPT feature extractor with BiLSTM temporal head for seizure detection.
    """
    
    def __init__(self, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        
        # EEGPT encoder (reuse existing wrapper; normalization already handled)
        self.eegpt = create_normalized_eegpt()
        self.eegpt.eval()
        for param in self.eegpt.parameters():
            param.requires_grad = False
        
        # BiLSTM temporal model
        self.lstm = nn.LSTM(
            input_size=2048,  # 4 summary tokens * 512 dim (flattened)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_windows, n_channels, n_samples) EEG windows
            
        Returns:
            probs: (batch, n_windows) seizure probabilities
        """
        batch_size, n_windows = x.shape[:2]
        
        # Extract EEGPT features for each window
        features = []
        for i in range(n_windows):
            window = x[:, i]
            with torch.no_grad():
                feat = self.eegpt.extract_features(window, summary=False)
                feat = feat.flatten(1)  # (batch, 2048)
            features.append(feat)
        
        features = torch.stack(features, dim=1)  # (batch, n_windows, 2048)
        
        # BiLSTM processing
        lstm_out, _ = self.lstm(features)  # (batch, n_windows, hidden*2)
        
        # Classification
        probs = self.classifier(lstm_out).squeeze(-1)  # (batch, n_windows)
        
        return probs
```

---

## 🎮 Post-Processing Implementation

### Advanced Post-Processor
```python
# src/brain_go_brrr/infra/eval/post_processing.py

import numpy as np
from scipy.ndimage import binary_closing, binary_opening
from typing import List, Tuple, Optional

class AdvancedPostProcessor:
    """
    Three-stage post-processing pipeline for temporal seizure detection.
    Critical for reducing false alarms and improving temporal alignment.
    """
    
    def __init__(
        self,
        hysteresis: Tuple[float, float] = (0.3, 0.7),
        merge_gap_sec: float = 2.0,
        min_duration_sec: float = 1.0,
        max_duration_sec: float = 600.0,
        fs: int = 256
    ):
        self.low_thresh, self.high_thresh = hysteresis
        self.merge_gap_samples = int(merge_gap_sec * fs)
        self.min_duration_samples = int(min_duration_sec * fs)
        self.max_duration_samples = int(max_duration_sec * fs)
        self.fs = fs
    
    def apply(self, probabilities: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Apply full post-processing pipeline.
        
        Args:
            probabilities: (n_samples,) raw model probabilities
            
        Returns:
            events: List of (start_sec, end_sec, confidence) tuples
        """
        # Stage 1: Hysteresis thresholding
        binary_mask = self._hysteresis_threshold(probabilities)
        
        # Stage 2: Morphological operations (gap merge)
        binary_mask = self._morphological_cleanup(binary_mask)
        
        # Stage 3: Extract and filter events
        events = self._extract_events(binary_mask, probabilities)
        
        return events
    
    def _hysteresis_threshold(self, probs: np.ndarray) -> np.ndarray:
        """Dual-threshold for stability."""
        mask = np.zeros_like(probs, dtype=bool)
        in_event = False
        
        for i, p in enumerate(probs):
            if not in_event and p > self.high_thresh:
                in_event = True
                mask[i] = True
            elif in_event:
                if p < self.low_thresh:
                    in_event = False
                else:
                    mask[i] = True
        
        return mask
    
    def _morphological_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Merge gaps and remove noise."""
        # Close gaps
        structure = np.ones(self.merge_gap_samples)
        mask = binary_closing(mask, structure=structure)
        
        # Remove small noise
        structure = np.ones(self.min_duration_samples // 2)
        mask = binary_opening(mask, structure=structure)
        
        return mask
    
    def _extract_events(
        self, 
        mask: np.ndarray, 
        probs: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """Extract events with duration filtering."""
        events = []
        
        # Find connected components
        in_event = False
        start_idx = 0
        
        for i, val in enumerate(mask):
            if val and not in_event:
                in_event = True
                start_idx = i
            elif not val and in_event:
                in_event = False
                duration = i - start_idx
                
                # Duration filtering
                if self.min_duration_samples <= duration <= self.max_duration_samples:
                    start_sec = start_idx / self.fs
                    end_sec = i / self.fs
                    confidence = float(np.mean(probs[start_idx:i]))
                    events.append((start_sec, end_sec, confidence))
        
        # Handle event extending to end
        if in_event:
            duration = len(mask) - start_idx
            if self.min_duration_samples <= duration <= self.max_duration_samples:
                start_sec = start_idx / self.fs
                end_sec = len(mask) / self.fs
                confidence = float(np.mean(probs[start_idx:]))
                events.append((start_sec, end_sec, confidence))
        
        return events
```

---

## 📏 NEDC Evaluation Integration

### NEDC Wrapper
```python
# src/brain_go_brrr/infra/eval/nedc_wrapper.py

import numpy as np
from typing import List, Tuple, Dict

# Note: For official reporting, add NEDC to PYTHONPATH externally, e.g.:
#   export PYTHONPATH=$PYTHONPATH:reference_repos/nedc_eeg_eval_v6.0.0/lib
# Then import and call the toolkit within this adapter.

class NEDCClinicalEvaluator:
    """
    Wrapper for NEDC Eval v6.0.0 to compute clinical metrics.
    This is what SeizureTransformer paper should have included!
    """
    
    def __init__(self):
        self.sensitivity_levels = [0.80, 0.85, 0.90, 0.95]
        
    def compute_all_metrics(
        self,
        predictions: List[Tuple[float, float, float]],
        ground_truth: List[Tuple[float, float]],
        duration_hours: float
    ) -> Dict[str, float]:
        """
        Compute comprehensive clinical metrics.
        
        Returns:
            Dictionary with FA/24h, TAES, ATWV, sensitivity, specificity
        """
        metrics = {}
        
        # Convert to NEDC format
        hyp_events = self._to_nedc_format(predictions)
        ref_events = self._to_nedc_format(ground_truth)
        
        # Compute TAES (Time-Aligned Event Scoring)
        taes_result = nedc.compute_taes(hyp_events, ref_events, jaccard_thresh=0.5)
        metrics['taes_sensitivity'] = taes_result['sensitivity']
        metrics['taes_precision'] = taes_result['precision']
        metrics['taes_f1'] = taes_result['f1']
        
        # Compute FA/24h at different sensitivity levels
        for sens_level in self.sensitivity_levels:
            filtered_preds = self._filter_by_sensitivity(predictions, sens_level)
            fa_count = self._count_false_alarms(filtered_preds, ground_truth)
            fa_per_24h = (fa_count / duration_hours) * 24
            metrics[f'fa_24h_at_{int(sens_level*100)}'] = fa_per_24h
        
        # Compute ATWV (Actual Term-Weighted Value)
        atwv = self._compute_atwv(predictions, ground_truth, beta=999.9)
        metrics['atwv'] = atwv
        
        # Epoch-based metrics
        epoch_result = nedc.compute_epoch(hyp_events, ref_events, epoch_sec=0.25)
        metrics['epoch_sensitivity'] = epoch_result['sensitivity']
        metrics['epoch_specificity'] = epoch_result['specificity']
        
        return metrics
    
    def _to_nedc_format(self, events: List[Tuple]) -> List[Dict]:
        """Convert events to NEDC expected format."""
        nedc_events = []
        for event in events:
            if len(event) == 3:
                start, end, conf = event
            else:
                start, end = event
                conf = 1.0
            
            nedc_events.append({
                'start': start,
                'end': end,
                'confidence': conf
            })
        
        return nedc_events
    
    def _compute_atwv(
        self,
        predictions: List[Tuple],
        ground_truth: List[Tuple],
        beta: float = 999.9
    ) -> float:
        """
        Compute Actual Term-Weighted Value.
        Beta = 999.9 is standard for seizure detection.
        """
        # Simplified ATWV calculation
        # Full implementation would use NIST F4DE toolkit
        
        n_ref = len(ground_truth)
        n_hyp = len(predictions)
        
        # Count correct detections (simplified - should use temporal overlap)
        n_correct = 0
        for pred in predictions:
            for ref in ground_truth:
                if self._events_overlap(pred, ref, min_overlap=0.5):
                    n_correct += 1
                    break
        
        # Calculate probabilities
        p_seizure = 0.08  # Approximate from TUSZ statistics
        p_correct_given_seizure = n_correct / n_ref if n_ref > 0 else 0
        p_fa_given_no_seizure = (n_hyp - n_correct) / (beta * n_ref) if n_ref > 0 else 0
        
        atwv = p_seizure * p_correct_given_seizure - beta * p_fa_given_no_seizure
        
        return max(0, atwv)  # ATWV can be negative
    
    def _events_overlap(
        self,
        event1: Tuple,
        event2: Tuple,
        min_overlap: float = 0.5
    ) -> bool:
        """Check if two events overlap sufficiently (Jaccard index)."""
        start1, end1 = event1[:2]
        start2, end2 = event2[:2]
        
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        
        jaccard = intersection / union if union > 0 else 0
        
        return jaccard >= min_overlap
```

---

## 🚀 Training Scripts

### SeizureTransformer Evaluation
```python
# experiments/tusz/evaluate_seizure_transformer.py

import numpy as np
from pathlib import Path
from brain_go_brrr.infra.data.tusz_detection_dataset import TUSZDetectionDataset
from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import SeizureTransformerWrapper
from brain_go_brrr.infra.eval.post_processing import AdvancedPostProcessor
from brain_go_brrr.infra.eval.nedc_wrapper import NEDCClinicalEvaluator

def main():
    """
    First-ever clinical evaluation of SeizureTransformer with proper metrics!
    """
    
    # Load model
    model = SeizureTransformerWrapper(
        weights_path=Path('data/models/pretrained/seizure_transformer_wu2025.pth')
    )
    
    # Load data
    dataset = TUSZDetectionDataset(
        root_dir=Path('data/datasets/tusz/v2.0.1'),
        split='dev'
    )
    
    # Initialize components
    post_processor = AdvancedPostProcessor(
        hysteresis=(0.3, 0.7),
        merge_gap_sec=2.0,
        min_duration_sec=1.0
    )
    evaluator = NEDCClinicalEvaluator()
    
    # Example: single-recording flow (replace with proper DataLoader/patient loop)
    patient_id = 'xxxx'  # placeholder
    eeg_data, ground_truth = dataset.load_recording(patient_id)
    probabilities = model.predict(eeg_data, apply_postprocessing=False)
    
    # Post-process
    predictions = post_processor.apply(probabilities)
    
    # Evaluate
    duration_hours = len(eeg_data[0]) / (256 * 3600)
    metrics = evaluator.compute_all_metrics(
        predictions, ground_truth, duration_hours
    )
    
    # Report (single recording)
    print("\n=== RECORDING RESULTS ===")
    print(f"FA/24h: {metrics['fa_24h']:.2f}")
    print(f"TAES F1: {metrics['taes_f1']:.3f}")
    
    # Save results
    import json
    with open('results/seizure_transformer_clinical_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nResults saved to results/seizure_transformer_clinical_metrics.json")
    print("Note: For official reporting, integrate NEDC evaluator outputs.")

if __name__ == '__main__':
    main()
```

### EEGPT + BiLSTM Training
```python
# experiments/tusz/train_eegpt_bilstm.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from brain_go_brrr.infra.ml_models.temporal_heads import EEGPTBiLSTM

def train_eegpt_bilstm():
    """
    Train BiLSTM temporal head on frozen EEGPT features.
    """
    
    # Model
    model = EEGPTBiLSTM(hidden_size=256, num_layers=2)
    
    # Loss and optimizer (only BiLSTM parameters)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    
    # Training loop
    for epoch in range(50):
        for batch in train_loader:
            # Forward pass
            predictions = model(batch['windows'])
            loss = criterion(predictions, batch['labels'])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        val_metrics = validate(model, val_loader)
        print(f"Epoch {epoch}: Val FA/24h: {val_metrics['fa_24h']:.2f}")
        
        # Early stopping based on FA/24h
        if val_metrics['fa_24h'] < best_fa:
            best_fa = val_metrics['fa_24h']
            torch.save(model.state_dict(), 'best_eegpt_bilstm.pth')
```

---

## 🧪 Testing & Validation

### Unit Tests
```python
# tests/unit/tusz/test_post_processing.py

import pytest
import numpy as np
from brain_go_brrr.infra.eval.post_processing import AdvancedPostProcessor

def test_hysteresis_thresholding():
    """Test dual-threshold stability."""
    processor = AdvancedPostProcessor(hysteresis=(0.3, 0.7))
    
    # Create synthetic probabilities with noise
    probs = np.array([0.1, 0.2, 0.8, 0.6, 0.4, 0.2, 0.1, 0.9, 0.8])
    
    events = processor.apply(probs)
    
    # Should detect two events (indices 2-5 and 7-8)
    assert len(events) == 2

def test_gap_merging():
    """Test that nearby events are merged."""
    processor = AdvancedPostProcessor(merge_gap_sec=2.0)
    
    # Events with 1-second gap (should merge)
    probs = np.zeros(1000)
    probs[100:200] = 0.9  # First event
    probs[256:356] = 0.9  # Second event (1 sec gap at 256Hz)
    
    events = processor.apply(probs)
    
    # Should merge into single event
    assert len(events) == 1
```

### Integration Tests
```python
# tests/integration/tusz/test_end_to_end.py

def test_seizure_transformer_pipeline():
    """Test complete pipeline from EEG to metrics."""
    
    # Load small test file
    test_eeg = load_test_recording()
    test_annotations = load_test_annotations()
    
    # Run pipeline
    model = SeizureTransformerWrapper()
    processor = AdvancedPostProcessor()
    evaluator = NEDCClinicalEvaluator()
    
    probs = model.predict(test_eeg)
    events = processor.apply(probs)
    metrics = evaluator.compute_all_metrics(events, test_annotations, 0.1)
    
    # Check metrics are computed
    assert 'fa_24h' in metrics
    assert 'taes_f1' in metrics
    assert 'atwv' in metrics
```

---

## 📊 Configuration Files

### SeizureTransformer Config
```yaml
# experiments/tusz/configs/seizure_transformer.yaml

model:
  name: seizure_transformer
  weights: data/models/pretrained/seizure_transformer_wu2025.pth
  window_sec: 60
  stride_sec: 30

preprocessing:
  resample_fs: 256
  bandpass: [0.5, 50]
  notch: 60
  normalize: z-score

postprocessing:
  hysteresis: [0.3, 0.7]
  merge_gap_sec: 2.0
  min_duration_sec: 1.0
  max_duration_sec: 600.0

evaluation:
  sensitivity_levels: [0.80, 0.85, 0.90, 0.95]
  taes_jaccard_thresh: 0.5
  atwv_beta: 0.1
  epoch_sec: 0.25
```

### EEGPT + BiLSTM Config
```yaml
# experiments/tusz/configs/eegpt_bilstm.yaml

model:
  name: eegpt_bilstm
  eegpt_checkpoint: data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt
  hidden_size: 256
  num_layers: 2
  dropout: 0.3

training:
  batch_size: 16
  learning_rate: 1e-3
  epochs: 50
  early_stopping_patience: 10
  gradient_clip: 1.0

data:
  window_sec: 4
  stride_sec: 2
  context_windows: 30  # 60 seconds of context
  augmentation:
    time_shift: 0.5
    amplitude_scale: [0.8, 1.2]
    gaussian_noise_snr: 20
```

---

## 🚀 CLI Commands

### Main CLI Interface
```python
# src/brain_go_brrr/cli/tusz_commands.py

import typer
from pathlib import Path

app = typer.Typer()

@app.command()
def evaluate(
    model: str = typer.Option("seizure_transformer", help="Model to evaluate"),
    data_dir: Path = typer.Option(..., help="TUSZ dataset directory"),
    output_dir: Path = typer.Option("results", help="Output directory"),
    config: Path = typer.Option(None, help="Config file")
):
    """Evaluate a model on TUSZ with clinical metrics."""
    
    if model == "seizure_transformer":
        from experiments.tusz.evaluate_seizure_transformer import main
        main(data_dir, output_dir, config)
    elif model == "eegpt_bilstm":
        from experiments.tusz.evaluate_eegpt_bilstm import main
        main(data_dir, output_dir, config)

@app.command()
def train(
    model: str = typer.Option("eegpt_bilstm", help="Model to train"),
    data_dir: Path = typer.Option(..., help="TUSZ dataset directory"),
    config: Path = typer.Option(None, help="Config file")
):
    """Train a temporal detection model."""
    
    if model == "eegpt_bilstm":
        from experiments.tusz.train_eegpt_bilstm import train_eegpt_bilstm
        train_eegpt_bilstm(data_dir, config)

# Usage:
# uv run bgb tusz evaluate --model seizure_transformer --data-dir data/datasets/tusz
# uv run bgb tusz train --model eegpt_bilstm --data-dir data/datasets/tusz
```

---

## 📈 Monitoring & Logging

### Experiment Tracking
```python
# Use MLflow or Weights & Biases for tracking

import mlflow
import mlflow.pytorch

mlflow.set_experiment("tusz_temporal_detection")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        "model": "seizure_transformer",
        "hysteresis_low": 0.3,
        "hysteresis_high": 0.7,
        "merge_gap_sec": 2.0
    })
    
    # Train/evaluate
    metrics = evaluate_model()
    
    # Log metrics
    mlflow.log_metrics({
        "fa_24h": metrics['fa_24h'],
        "taes_f1": metrics['taes_f1']
    })
    
    # Log model
    mlflow.pytorch.log_model(model, "model")
```

---

## 🎯 Deliverables Checklist

### Week 1 Deliverables
- [ ] NEDC Eval integrated and working
- [ ] SeizureTransformer wrapper complete
- [ ] Post-processing pipeline implemented
- [ ] First clinical metrics computed
- [ ] Results JSON with FA/24h, TAES, ATWV

### Week 2 Deliverables
- [ ] EEGPT + BiLSTM trained
- [ ] Comparative analysis complete
- [ ] Hyperparameter tuning done
- [ ] Publication draft started
- [ ] Code released on GitHub

### Success Criteria
- [ ] FA/24h < 10 at 95% sensitivity (stretch: < 5)
- [ ] TAES F1 > 0.5
- [ ] ATWV > 0.5
- [ ] Reproducible results with seeds
- [ ] Documentation complete

---

## 🔧 Troubleshooting

## ⚖️ Clinical Threshold Policy & Freezing

- Select a single operating threshold θ on the validation (dev) set to meet the target sensitivity (e.g., 0.90).
- Freeze θ for all test-set evaluations; do not retune on test.
- Report FA/24h at Sens ∈ {0.80, 0.85, 0.90, 0.95} and provide full TAES curves.
- Document θ, post-processing params (`max_gap`, `min_event_dur`, hysteresis) with metrics for reproducibility.

## 🧪 Reproducibility & Checkpointing

- Seeds: Set and log Python/Numpy/Torch/CUDA seeds; enable deterministic flags where feasible.
- Checkpoints: Save model, optimizer, scheduler states, and RNG seeds per epoch; support resume-safe schedulers.
- Paths: Default `SEIZURE_TRANSFORMER_CKPT=data/models/pretrained/seizure_transformer_wu2025.pth` (override via env var).
- Schema tests: Validate hypothesis CSV/XML schema against the NEDC evaluator before large runs.

### Common Issues

**Issue**: NEDC Eval import errors
```bash
# Solution: Add to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:reference_repos/nedc_eeg_eval_v6.0.0/lib
```

**Issue**: SeizureTransformer weights not loading
```python
# Solution: Check PyTorch version compatibility
# May need to use weights_only=False for older checkpoints
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
```

**Issue**: High FA/24h rate
```python
# Solution: Tune post-processing more aggressively
processor = AdvancedPostProcessor(
    hysteresis=(0.4, 0.8),  # Higher thresholds
    merge_gap_sec=5.0,      # Merge more gaps
    min_duration_sec=2.0    # Filter more short events
)
```

---

**THIS IMPLEMENTATION GUIDE IS YOUR EXECUTION BLUEPRINT**

Ready to implement? Start with Phase 1: Infrastructure setup!
