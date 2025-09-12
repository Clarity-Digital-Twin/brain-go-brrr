# 🎯 INTENDED SEIZURE TRANSFORMER APPLICATION - IMPLEMENTATION PLAN

**Status**: READY FOR SENIOR REVIEW  
**Created**: December 12, 2024  
**Updated**: December 12, 2024  
**Purpose**: Explicit, actionable plan to bridge CURRENT → IDEAL implementation

---

## 📊 EXECUTIVE SUMMARY

### Current State (BROKEN)
- ❌ TSE parser accepts ANY 2-field line (not just seizures)
- ❌ Training bypasses critical preprocessing (no bandpass/notch)
- ❌ Wrong supervision (window labels expanded to timesteps)
- ❌ External dependency on wu_2025 package
- ❌ No evaluation script for pretrained weights

### Target State (IDEAL)
- ✅ Correct TSE parsing (seizures only)
- ✅ Full preprocessing pipeline (z-score → resample → bandpass → notch)
- ✅ Proper per-timestep supervision
- ✅ Self-contained model in src/
- ✅ Evaluation achieving AUROC ~0.876 on TUSZ test

### Critical Path
1. **Fix data corruption** (TSE parser) - HIGHEST PRIORITY
2. **Copy model architecture** to src/ 
3. **Implement preprocessing** exactly as OSS
4. **Create evaluation script** for pretrained weights
5. **Validate AUROC** matches paper

---

## 🚨 PRIORITY 1: FIX DATA CORRUPTION (TSE Parser Bug)

### THE BUG (CURRENT)
```python
# src/brain_go_brrr/infra/data/tusz_detection_dataset.py - LINE ~147
def _parse_tse(self, tse_file: Path) -> List[Tuple[float, float, str]]:
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:  # ❌ ACCEPTS ANY 2-NUMBER LINE!
            start = float(parts[0])
            end = float(parts[1])
            label = parts[2] if len(parts) > 2 else ""
            
            # This accepts background/artifacts as seizures!
            if "seiz" in label or len(parts) >= 2:  # ❌ BROKEN LOGIC
                annotations.append((start, end, label))
```

### THE FIX (IDEAL)
```python
def _parse_tse(self, tse_file: Path) -> List[Tuple[float, float, str]]:
    """Parse TSE file for seizure annotations ONLY.
    
    TSE format:
    - start_time end_time [label]
    - Only lines with 'seiz' in label are seizures
    - Background/artifact/other labels should be ignored
    """
    annotations = []
    with open(tse_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 2:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2] if len(parts) > 2 else ""
                    
                    # ✅ ONLY accept seizure annotations
                    if "seiz" in label.lower():
                        annotations.append((start, end, label))
                        logger.debug(f"TSE {tse_file.name}:{line_num} - Seizure: {start:.2f}-{end:.2f}s ({label})")
                    else:
                        logger.debug(f"TSE {tse_file.name}:{line_num} - Skipping non-seizure: {label}")
                        
                except ValueError as e:
                    logger.warning(f"TSE {tse_file.name}:{line_num} - Parse error: {e}")
                    continue
    
    logger.info(f"Parsed {len(annotations)} seizure annotations from {tse_file.name}")
    return annotations
```

### VERIFICATION TEST
```python
def test_tse_parser_only_accepts_seizures():
    """Ensure TSE parser correctly filters annotations."""
    tse_content = """
    # Test TSE file
    0.0 10.0 background
    10.0 20.0 seizure_type_1
    20.0 30.0 artifact
    30.0 40.0 focal_seizure
    40.0 50.0 
    """
    
    # Should only return the two seizure lines
    annotations = _parse_tse(tse_content)
    assert len(annotations) == 2
    assert annotations[0] == (10.0, 20.0, "seizure_type_1")
    assert annotations[1] == (30.0, 40.0, "focal_seizure")
```

---

## 🔧 PRIORITY 2: MIGRATE MODEL TO SRC

### CURRENT (BROKEN)
```python
# experiments/seizure_transformer/train_tusz.py
from wu_2025.architecture import SeizureTransformer  # ❌ External dependency
```

### ACTION STEPS
```bash
# 1. Copy the architecture file
cp reference_repos/SeizureTransformer/wu_2025/src/wu_2025/architecture.py \
   src/brain_go_brrr/infra/ml_models/seizure_transformer.py

# 2. Update imports in the new file
# In seizure_transformer.py, change:
# from wu_2025.utils import whatever
# To:
# from brain_go_brrr.infra.ml_models.seizure_transformer_utils import whatever

# 3. Remove external package
uv remove wu_2025

# 4. Update all imports project-wide
# Find all: from wu_2025.architecture import SeizureTransformer
# Replace: from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer
```

### VERIFICATION
```python
# Test that model loads correctly
from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer

model = SeizureTransformer()  # Default params
assert model.in_channels == 19
assert model.in_samples == 15360
print("✅ Model migrated successfully")
```

---

## 🧪 PRIORITY 3: IMPLEMENT EXACT PREPROCESSING

### CURRENT (MISSING)
```python
# src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py
def predict(self, ...):
    # ❌ NO PREPROCESSING - goes straight to model!
    x = torch.from_numpy(eeg_data)
    output = self.model(x)
```

### IMPLEMENTATION
```python
# src/brain_go_brrr/infra/ml_models/seizure_transformer_utils.py

import numpy as np
from scipy.signal import butter, lfilter, iirnotch, resample

class SeizurePreprocessor:
    """Exact preprocessing from Wu et al. 2025 OSS implementation.
    
    CRITICAL: This preprocessing is REQUIRED for using pretrained weights!
    Different preprocessing = degraded performance.
    
    Pipeline (exact order):
    1. Z-score normalization (per-channel, over full recording)
    2. Resample to 256Hz if needed
    3. Bandpass 0.5-120Hz (order=3, causal)
    4. Notch filters at 1Hz and 60Hz
    """
    
    def __init__(self):
        # Pre-compute filter coefficients at 256Hz
        self.fs = 256
        self.lowcut = 0.5
        self.highcut = 120  # Note: 120Hz, not 100Hz!
        
        # Notch filters (Q=30 from OSS)
        self.notch_1_b, self.notch_1_a = iirnotch(1, Q=30, fs=256)
        self.notch_60_b, self.notch_60_a = iirnotch(60, Q=30, fs=256)
        
        # Bandpass coefficients
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        self.bp_b, self.bp_a = butter(3, [low, high], btype='band')
    
    def preprocess(self, eeg: np.ndarray, fs_original: int) -> np.ndarray:
        """
        Apply exact preprocessing from Wu et al. 2025.
        
        Args:
            eeg: Raw EEG data (n_channels, n_samples)
            fs_original: Original sampling rate
            
        Returns:
            Preprocessed EEG (n_channels, n_samples_resampled)
        """
        # 1. Z-score normalization (per-channel, over full recording)
        # CRITICAL: This is done BEFORE windowing in the OSS code!
        mean = np.mean(eeg, axis=1, keepdims=True)
        std = np.std(eeg, axis=1, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        eeg = (eeg - mean) / std
        
        # 2. Resample to 256Hz if needed
        if fs_original != 256:
            n_samples_new = int(eeg.shape[1] * 256.0 / fs_original)
            eeg_resampled = np.zeros((eeg.shape[0], n_samples_new))
            for ch in range(eeg.shape[0]):
                eeg_resampled[ch] = resample(eeg[ch], n_samples_new)
            eeg = eeg_resampled
        
        # 3. Bandpass filter (0.5-120Hz, order=3, causal)
        # CRITICAL: Use lfilter (causal), not filtfilt (zero-phase)!
        for ch in range(eeg.shape[0]):
            eeg[ch] = lfilter(self.bp_b, self.bp_a, eeg[ch])
        
        # 4. Notch filters (1Hz, 60Hz)
        for ch in range(eeg.shape[0]):
            eeg[ch] = lfilter(self.notch_1_b, self.notch_1_a, eeg[ch])
            eeg[ch] = lfilter(self.notch_60_b, self.notch_60_a, eeg[ch])
        
        return eeg
```

### POST-PROCESSING IMPLEMENTATION
```python
# src/brain_go_brrr/infra/ml_models/seizure_transformer_utils.py

from scipy.ndimage import binary_opening, binary_closing
import numpy as np

class SeizurePostProcessor:
    """Exact post-processing from Wu et al. 2025 OSS implementation."""
    
    def __init__(
        self,
        threshold: float = 0.8,
        morph_open_size: int = 5,
        morph_close_size: int = 5,
        min_duration_sec: float = 2.0,
        fs: int = 256,
    ):
        self.threshold = threshold
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        self.min_duration_sec = min_duration_sec
        self.fs = fs
        self.min_duration_samples = int(min_duration_sec * fs)
    
    def postprocess(self, probs: np.ndarray) -> np.ndarray:
        """Apply exact OSS post-processing pipeline."""
        # 1. Threshold at 0.8
        binary = (probs > self.threshold).astype(int)
        
        # 2. Morphological opening (remove short bursts)
        structure = np.ones(self.morph_open_size)
        binary = binary_opening(binary, structure=structure)
        
        # 3. Morphological closing (fill gaps)
        structure = np.ones(self.morph_close_size)
        binary = binary_closing(binary, structure=structure)
        
        # 4. Remove events < 2 seconds
        binary = self._remove_short_events(binary)
        
        return binary.astype(int)
    
    def _remove_short_events(self, binary: np.ndarray) -> np.ndarray:
        """Remove seizure events shorter than min_duration."""
        # Find connected components (seizure events)
        from scipy.ndimage import label
        labeled, num_features = label(binary)
        
        # Check each event's duration
        for i in range(1, num_features + 1):
            event_mask = labeled == i
            event_length = np.sum(event_mask)
            if event_length < self.min_duration_samples:
                binary[event_mask] = 0
        
        return binary
```

### INTEGRATION INTO WRAPPER (COMPLETE)
```python
# src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py

import torch
import numpy as np
import mne
from pathlib import Path
from typing import Optional

class SeizureTransformerWrapper:
    """Production wrapper for SeizureTransformer with exact OSS preprocessing."""
    
    REQUIRED_CHANNELS = 19  # Model requirement
    WINDOW_SIZE_SEC = 60.0  # 60 second windows
    WINDOW_SIZE_SAMPLES = 15360  # 60s * 256Hz
    
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load model with DEFAULT params (CRITICAL!)
        self.model = SeizureTransformer()  # No parameters!
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)  # nosec:weights_only
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()  # CRITICAL: Disable dropout
        
        # Initialize processors
        self.preprocessor = SeizurePreprocessor()
        self.postprocessor = SeizurePostProcessor()
    
    def predict(self, raw: mne.io.Raw) -> np.ndarray:
        """Run inference on raw EEG data with full pipeline."""
        # 1. Validate montage
        if not self._is_unipolar(raw):
            raise ValueError("Model requires unipolar/referential montage")
        
        # 2. Get data and ensure 19 channels
        data = self._prepare_channels(raw)  # (19, n_samples)
        fs = int(raw.info['sfreq'])
        
        # 3. Apply preprocessing
        data = self.preprocessor.preprocess(data, fs_original=fs)
        
        # 4. Create windows (60s, no overlap for inference)
        windows = self._create_windows(data)
        
        # 5. Run inference
        all_preds = []
        with torch.no_grad():
            for window in windows:
                # Ensure correct shape (batch=1, channels=19, samples=15360)
                x = torch.from_numpy(window).float()
                x = x.unsqueeze(0).to(self.device)
                
                # Forward pass
                output = self.model(x)  # (1, 15360)
                probs = torch.sigmoid(output)
                all_preds.append(probs.cpu().numpy().squeeze())
        
        # 6. Concatenate and trim to original length
        if len(all_preds) > 0:
            y_pred = np.concatenate(all_preds)
            # Trim to original sequence length if needed
            original_len = data.shape[1]
            y_pred = y_pred[:original_len]
        else:
            y_pred = np.array([])
        
        # 7. Apply post-processing
        y_pred = self.postprocessor.postprocess(y_pred)
        
        return y_pred
    
    def _is_unipolar(self, raw: mne.io.Raw) -> bool:
        """Check if montage is unipolar/referential."""
        # Check for bipolar channel names (contain '-')
        for ch_name in raw.ch_names:
            if '-' in ch_name and not ch_name.startswith('EEG'):
                return False  # Likely bipolar
        return True
    
    def _prepare_channels(self, raw: mne.io.Raw) -> np.ndarray:
        """Ensure exactly 19 channels, padding with zeros if needed."""
        data = raw.get_data()
        n_channels, n_samples = data.shape
        
        if n_channels == self.REQUIRED_CHANNELS:
            return data
        elif n_channels > self.REQUIRED_CHANNELS:
            # Take first 19 channels
            return data[:self.REQUIRED_CHANNELS, :]
        else:
            # Pad with zeros
            padded = np.zeros((self.REQUIRED_CHANNELS, n_samples))
            padded[:n_channels, :] = data
            return padded
    
    def _create_windows(self, data: np.ndarray) -> list:
        """Create 60-second windows with no overlap for inference."""
        n_channels, n_samples = data.shape
        windows = []
        
        # Calculate number of complete windows
        n_windows = n_samples // self.WINDOW_SIZE_SAMPLES
        
        for i in range(n_windows):
            start = i * self.WINDOW_SIZE_SAMPLES
            end = start + self.WINDOW_SIZE_SAMPLES
            window = data[:, start:end]
            windows.append(window)
        
        # Handle last partial window with zero-padding
        remaining = n_samples % self.WINDOW_SIZE_SAMPLES
        if remaining > 0:
            last_window = np.zeros((n_channels, self.WINDOW_SIZE_SAMPLES))
            last_window[:, :remaining] = data[:, -remaining:]
            windows.append(last_window)
        
        return windows
```

---

## 📊 PRIORITY 3.5: ENHANCE DATASET LOADER

### DATASET INTEGRATION
```python
# src/brain_go_brrr/infra/data/tusz_detection_dataset.py (ENHANCED)

class TUSZDetectionDataset:
    """Enhanced dataset with proper preprocessing integration."""
    
    def __init__(
        self,
        root_dir: Path,
        split: str,
        cfg: WindowConfig,
        preprocessor: Optional[SeizurePreprocessor] = None,
        ensure_unipolar: bool = True,
        max_windows: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.split = split
        self.cfg = cfg
        self.preprocessor = preprocessor or SeizurePreprocessor()
        self.ensure_unipolar = ensure_unipolar
        self.max_windows = max_windows
        
        # Build index of windows
        self.index = self._build_index()
    
    def __getitem__(self, idx):
        """Get preprocessed window with label."""
        window_info = self.index[idx]
        
        # Load EDF
        raw = mne.io.read_raw_edf(window_info['edf_path'], preload=True, verbose=False)
        
        # Validate montage
        if self.ensure_unipolar:
            if not self._is_unipolar(raw):
                raise ValueError(f"Non-unipolar montage in {window_info['edf_path']}")
        
        # Get data in Volts
        data = raw.get_data()  # (n_channels, n_samples)
        fs = int(raw.info['sfreq'])
        
        # Apply preprocessing (CRITICAL: before windowing!)
        data = self.preprocessor.preprocess(data, fs_original=fs)
        
        # Extract window
        start_sample = window_info['start_sample']
        end_sample = window_info['end_sample']
        window = data[:, start_sample:end_sample]
        
        # Ensure 19 channels
        window = self._ensure_channels(window)
        
        # Get label (per-timestep for training)
        label = self._get_window_label(window_info)
        
        return torch.from_numpy(window).float(), torch.from_numpy(label).float()
    
    def _ensure_channels(self, data: np.ndarray) -> np.ndarray:
        """Ensure exactly 19 channels."""
        n_channels = data.shape[0]
        if n_channels == 19:
            return data
        elif n_channels > 19:
            return data[:19, :]
        else:
            # Pad with zeros
            padded = np.zeros((19, data.shape[1]))
            padded[:n_channels, :] = data
            return padded
    
    def _get_window_label(self, window_info) -> np.ndarray:
        """Get per-timestep labels for window."""
        # For inference: return single label
        if self.split == "eval":
            return np.array([window_info['has_seizure']], dtype=np.float32)
        
        # For training: return per-timestep labels
        labels = np.zeros(self.cfg.window_samples, dtype=np.float32)
        for start, end in window_info['seizure_segments']:
            labels[start:end] = 1.0
        return labels
```

## 🎯 PRIORITY 4: CREATE EVALUATION SCRIPT

### PURPOSE
Validate pretrained weights achieve paper's AUROC on TUSZ test set

### IMPLEMENTATION
```python
# scripts/evaluate_seizure_transformer.py

import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import torch
from tqdm import tqdm

from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer
from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import SeizureTransformerWrapper
from brain_go_brrr.infra.data.tusz_detection_dataset import TUSZDetectionDataset

def evaluate_pretrained_model():
    """Evaluate pretrained SeizureTransformer on TUSZ test set.
    
    Expected Results (from paper):
    - AUROC: 0.876 ± 0.02
    - FA/24h: ≤5 at 90% sensitivity
    - Processing: ~4s per hour of EEG
    """
    print("=" * 60)
    print("SEIZURE TRANSFORMER EVALUATION")
    print("Using pretrained weights on TUSZ eval set")
    print("=" * 60)
    
    # 1. Load pretrained model
    model_path = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/models/pretrained/seizure_transformer_wu2025.pth")
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at {model_path}")
    
    wrapper = SeizureTransformerWrapper(model_path)
    print(f"✅ Loaded pretrained weights from {model_path}")
    
    # 2. Create test dataset (TUSZ eval split ONLY!)
    tusz_root = Path("/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/datasets/tusz/edf")
    test_dataset = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="eval",  # CRITICAL: Use eval split only!
        cfg=WindowConfig(
            fs=256,
            window_sec=60.0,
            stride_sec=60.0,  # No overlap for inference
        ),
        preprocessor=wrapper.preprocessor,  # Use exact preprocessing
        ensure_unipolar=True,
    )
    print(f"✅ Loaded TUSZ eval set: {len(test_dataset)} windows")
    
    # 3. Run inference
    all_preds = []
    all_labels = []
    
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print("\nRunning inference...")
    for batch_data, batch_labels in tqdm(dataloader):
        with torch.no_grad():
            # Move to device
            batch_data = batch_data.to(wrapper.device)
            
            # Run model
            outputs = wrapper.model(batch_data)  # (batch, 15360)
            
            # Apply sigmoid (model outputs logits)
            probs = torch.sigmoid(outputs)
            
            # Store predictions
            all_preds.extend(probs.cpu().numpy().flatten())
            all_labels.extend(batch_labels.numpy().flatten())
    
    # 4. Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Apply post-processing
    all_preds_binary = wrapper.postprocessor.postprocess(all_preds)
    
    # Calculate AUROC (on probabilities, before post-processing)
    auroc = roc_auc_score(all_labels, all_preds)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"AUROC: {auroc:.3f} (Expected: 0.876)")
    print(f"Total samples: {len(all_labels):,}")
    print(f"Positive samples: {all_labels.sum():,} ({100*all_labels.mean():.1f}%)")
    
    # Success check
    if abs(auroc - 0.876) < 0.02:
        print("\n✅ SUCCESS: AUROC matches paper!")
    else:
        print(f"\n⚠️ WARNING: AUROC differs from paper by {abs(auroc - 0.876):.3f}")
    
    return auroc

if __name__ == "__main__":
    evaluate_pretrained_model()
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Critical Fixes (Day 1)
- [ ] Fix TSE parser bug (only accept seizure annotations)
- [ ] Write unit test for TSE parser
- [ ] Copy model architecture to src/
- [ ] Remove wu_2025 dependency
- [ ] Update all imports

### Phase 2: Preprocessing Pipeline (Day 2)
- [ ] Implement SeizurePreprocessor class
- [ ] Add preprocessing to wrapper
- [ ] Write preprocessing unit tests
- [ ] Verify filter coefficients match OSS

### Phase 3: Evaluation (Day 3)
- [ ] Create evaluation script
- [ ] Download pretrained weights if needed
- [ ] Run on TUSZ eval set
- [ ] Verify AUROC ~0.876
- [ ] Document results

### Phase 4: Production Wrapper (Day 4-5)
- [ ] Implement full SeizureTransformerWrapper
- [ ] Add post-processing pipeline
- [ ] Create CLI interface
- [ ] Add batch processing support
- [ ] Write integration tests

### Phase 5: Clinical API (Week 2)
- [ ] Create FastAPI endpoints
- [ ] Add streaming support
- [ ] Implement caching
- [ ] Add monitoring/logging
- [ ] Deploy to staging

---

## 🚨 CRITICAL SUCCESS FACTORS

### 1. **USE PRETRAINED WEIGHTS**
```python
# NEVER train from scratch!
model = SeizureTransformer()
checkpoint = torch.load("seizure_transformer_wu2025.pth")
model.load_state_dict(checkpoint)
model.eval()  # ALWAYS use eval mode
```

### 2. **EXACT PREPROCESSING**
- Must use Wu's preprocessing for pretrained weights
- Different preprocessing = degraded performance
- Order matters: z-score → resample → bandpass → notch

### 3. **CORRECT DATA SPLIT**
```python
# ONLY use TUSZ eval/ for testing
split="eval"  # NOT "train" or "dev"
```

### 4. **UNIPOLAR MONTAGE**
- Model REQUIRES unipolar/referential montage
- Check and reject bipolar montages
- Exactly 19 channels required

### 5. **POST-PROCESSING PARAMETERS**
- Threshold: 0.8 (hardcoded)
- Morphological kernels: 5 (hardcoded)
- Min duration: 2.0 seconds (hardcoded)

---

## 📊 EXPECTED OUTCOMES

### After Implementation
- ✅ AUROC: 0.876 on TUSZ test set
- ✅ Processing: <4s per hour of EEG
- ✅ FA/24h: ≤5 at 90% sensitivity
- ✅ Production-ready seizure detection

### Deliverables
1. Fixed TSE parser with tests
2. Self-contained model in src/
3. Working preprocessing pipeline
4. Evaluation script showing paper parity
5. Production wrapper for clinical use

---

## 🔴 RISKS & MITIGATIONS

| Risk | Impact | Mitigation |
|------|--------|------------|
| Weights file corrupted | Can't reproduce | Verify checksum, re-download |
| Channel order mismatch | Poor performance | Test multiple orderings |
| Preprocessing differences | Degraded AUROC | Match OSS exactly |
| Memory issues (19ch × 15360) | Can't run inference | Use smaller batches |
| Montage incompatibility | Can't process files | Add montage converter |

---

## 📝 NOTES FOR SENIOR REVIEW

### ✅ CONFIRMED ASSETS IN PLACE:
1. **Pretrained weights**: ✅ AVAILABLE at `/data/models/pretrained/seizure_transformer_wu2025.pth` (169MB)
2. **TUSZ eval dataset**: ✅ AVAILABLE at `/data/datasets/tusz/edf/eval/` (865 EDF files)
3. **EEGPT weights**: ✅ AVAILABLE at `/data/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt`

### Questions Requiring Decisions:
1. **Channel ordering**: Not specified in paper - may need experimentation
2. **Production deployment**: Docker? Kubernetes? Cloud?
3. **Clinical validation**: Which hospital data to test on?
4. **Regulatory considerations**: FDA pathway needed?

### Resource Requirements:
- **Compute**: GPU preferred (but CPU works)
- **Storage**: ~50GB for TUSZ test set
- **Time**: 5 days for full implementation
- **Testing**: Additional week for validation

---

## ✅ APPROVAL CHECKPOINT

**This plan is ready for senior review and approval.**

Once approved, we can begin implementation immediately with Phase 1 (Critical Fixes).

Expected timeline: 
- Week 1: Phases 1-4 (Core implementation)
- Week 2: Phase 5 (Clinical API)
- Week 3: Testing and validation
- Week 4: Production deployment

---

**END OF INTENDED IMPLEMENTATION PLAN**

*Document prepared for senior technical review*  
*Questions? Contact the implementation team*