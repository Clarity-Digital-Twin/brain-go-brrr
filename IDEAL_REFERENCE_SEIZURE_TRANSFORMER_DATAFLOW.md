# 🟢 IDEAL SEIZURE TRANSFORMER DATAFLOW (What We Should Build)

**Status**: PROPOSED ARCHITECTURE
**Created**: December 12, 2024
**Purpose**: Document the CORRECT implementation based on OSS reference

---

## ✅ Correct Architecture Pattern

### 1. WHERE THE MODEL SHOULD LIVE
```
src/brain_go_brrr/infra/ml_models/
├── seizure_transformer.py      # COPIED architecture (NOT imported from reference)
├── seizure_transformer_wrapper.py  # Our wrapper with preprocessing/postprocessing
└── seizure_transformer_utils.py    # Preprocessing & postprocessing utilities
```

**PRINCIPLE**: Reference repos are for LEARNING, not IMPORTING. Copy the architecture, understand it, own it.

### 2. PREPROCESSING PIPELINE (From OSS)
```python
# src/brain_go_brrr/infra/ml_models/seizure_transformer_utils.py

class SeizurePreprocessor:
    """Exact preprocessing from Wu et al. 2025."""
    
    def __init__(self, fs: int = 256):
        self.fs = fs
        self.lowcut = 0.5
        self.highcut = 120
        # Pre-compute filter coefficients (ALWAYS for 256Hz after resampling)
        self.notch_1_b, self.notch_1_a = iirnotch(1, Q=30, fs=256)
        self.notch_60_b, self.notch_60_a = iirnotch(60, Q=30, fs=256)
    
    def preprocess(self, eeg: np.ndarray, fs_original: int) -> np.ndarray:
        """Apply OSS preprocessing pipeline (EXACT order from implementation).
        
        NOTE: Preprocessing happens in TWO stages:
        1. In get_dataloader: Z-score normalization + Resampling
        2. In preprocess_clip: Bandpass + Notch filters
        
        Order:
        1. Z-score normalization (per-channel, computed over the entire recording before windowing)
        2. Resample to 256Hz if needed
        3. Bandpass filter (0.5-120Hz, order=3, using lfilter not filtfilt)
        4. Notch filters (1Hz, 60Hz)
        """
        # Z-score normalize (per-channel over full sequence, not window-wise)
        eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / np.std(eeg, axis=1, keepdims=True)
        
        # Resample if needed
        if fs_original != 256:
            new_n_samples = int(eeg.shape[1] * 256.0 / fs_original)
            eeg = resample(eeg, new_n_samples, axis=1)
        
        # Bandpass filter (order=3 specifically, causal filtering)
        nyq = 0.5 * 256
        low = self.lowcut / nyq
        high = self.highcut / nyq
        b, a = butter(3, [low, high], btype='band')
        eeg = lfilter(b, a, eeg)
        
        # Notch filters
        eeg = lfilter(self.notch_1_b, self.notch_1_a, eeg)
        eeg = lfilter(self.notch_60_b, self.notch_60_a, eeg)
        
        return eeg
```

### 3. DATASET LOADER (Enhanced)
```python
# src/brain_go_brrr/infra/data/tusz_detection_dataset.py

class TUSZDetectionDataset:
    def __init__(
        self,
        root_dir: Path,
        split: str,
        cfg: WindowConfig,
        preprocessor: SeizurePreprocessor | None = None,  # ADD THIS
        ensure_unipolar: bool = True,  # ADD THIS
    ):
        self.preprocessor = preprocessor or SeizurePreprocessor()
        self.ensure_unipolar = ensure_unipolar
        
    def __getitem__(self, idx):
        # Load raw EDF
        # NOTE (OSS parity): The reference implementation uses
        # epilepsy2bids.Eeg.loadEdfAutoDetectMontage and asserts UNIPOLAR montage.
        # If using MNE here, enforce equivalent constraints and 19-channel referential data.
        raw = mne.io.read_raw_edf(edf_path, preload=True)
        
        # CRITICAL: Check montage
        if self.ensure_unipolar:
            if not self._is_unipolar(raw):
                raise ValueError(f"Non-unipolar montage detected in {edf_path}")
        
        # Get data in Volts
        data = raw.get_data()  # (channels, samples)
        
        # Apply preprocessing
        fs_original = int(raw.info['sfreq'])  # source sampling rate
        data = self.preprocessor.preprocess(data, fs_original=fs_original)
        
        # Window extraction...
        return torch.from_numpy(data), label
```

### 4. POST-PROCESSING (Match OSS Exactly)
```python
# src/brain_go_brrr/infra/eval/seizure_postprocessor.py

class SeizurePostProcessor:
    """Exact post-processing from Wu et al. 2025."""
    
    def __init__(
        self,
        threshold: float = 0.8,  # OSS default
        morph_open_size: int = 5,  # OSS default
        morph_close_size: int = 5,  # OSS default
        min_duration_sec: float = 2.0,  # OSS default
        fs: int = 256,
    ):
        self.threshold = threshold
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        self.min_duration_sec = min_duration_sec
        self.fs = fs
    
    def postprocess(self, probs: np.ndarray) -> np.ndarray:
        """Apply OSS post-processing pipeline.
        
        1. Threshold at 0.8
        2. Morphological opening (remove short bursts)
        3. Morphological closing (fill gaps)
        4. Remove events < 2 seconds
        """
        # Threshold
        binary = (probs > self.threshold).astype(int)
        
        # Morphological opening
        binary = morphological_filter_1d(binary, "opening", self.morph_open_size)
        
        # Morphological closing
        binary = morphological_filter_1d(binary, "closing", self.morph_close_size)
        
        # Remove short events
        binary = remove_short_events(binary, self.min_duration_sec, self.fs)
        
        return binary
```

### 5. MODEL WRAPPER (Complete Integration)
```python
# src/brain_go_brrr/infra/ml_models/seizure_transformer_wrapper.py

class SeizureTransformerWrapper:
    """Complete wrapper with all OSS components."""
    
    def __init__(
        self,
        model_path: Path | None = None,
        device: str = "cuda",
    ):
        # Load model architecture (OUR copy, not reference)
        from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer
        self.model = SeizureTransformer(
            in_channels=19,
            in_samples=15360,
            drop_rate=0.1,
        )
        
        # Load weights if provided
        if model_path and model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize processors
        self.preprocessor = SeizurePreprocessor(fs=256)  # Note: fs param ignored for notch filters
        self.postprocessor = SeizurePostProcessor()  # All params hardcoded
    
    def predict(
        self,
        eeg: np.ndarray,
        fs_original: int,
        apply_preprocessing: bool = True,
        apply_postprocessing: bool = True,
    ) -> np.ndarray:
        """Full prediction pipeline matching OSS."""
        
        # Preprocessing
        if apply_preprocessing:
            eeg = self.preprocessor.preprocess(eeg, fs_original=fs_original)
        
        # Sliding windows with proper overlap
        windows = self._create_windows(eeg)  # 60s @ 256Hz (15360 samples), 0% overlap for inference
        
        # Model inference
        predictions = []
        with torch.no_grad():
            for window in windows:
                x = torch.from_numpy(window).unsqueeze(0).to(self.device)
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
        
        # Concatenate predictions
        y_pred = np.concatenate(predictions).flatten()
        
        # Post-processing
        if apply_postprocessing:
            y_pred = self.postprocessor.postprocess(y_pred)
        
        return y_pred
```

### 6. TRAINING SCRIPT (Clean Import)
```python
# experiments/seizure_transformer/train_tusz.py

# CORRECT: Import from OUR codebase, not reference
from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer
from brain_go_brrr.infra.ml_models.seizure_transformer_utils import SeizurePreprocessor
from brain_go_brrr.infra.data.tusz_detection_dataset import TUSZDetectionDataset
from brain_go_brrr.infra.eval.seizure_postprocessor import SeizurePostProcessor
from brain_go_brrr.infra.eval.nedc_metrics import NEDCEvaluator

def train():
    # Initialize components
    preprocessor = SeizurePreprocessor(fs=256)
    
    # Create dataset with preprocessing
    train_ds = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="train",
        cfg=WindowConfig(
            fs=256,
            window_sec=60.0,
            stride_sec=15.0,  # 75% overlap for training (from paper)
            # For inference: stride_sec=60.0 (0% overlap)
        ),
        preprocessor=preprocessor,
        ensure_unipolar=True,
    )
    
    # Model from OUR codebase
    model = SeizureTransformer(
        in_channels=19,
        in_samples=15360,
        drop_rate=0.1,
    )
    
    # Training parameters (from paper Page 3):
    # - Optimizer: RAdam
    # - Learning rate: 1e-3
    # - Weight decay: 2e-5
    # - Batch size: 256
    # - Drop rate: 0.1
    # - Loss: Binary Cross-Entropy
    # - Early stopping: 12 epochs patience
    # - Max epochs: 100
    # 
    # Data balancing (from paper):
    # D = Dps ∪ D*fs ∪ D*ns where:
    # - |D*fs| = 0.3 × |Dps| (30% of partial-seizure windows)
    # - |D*ns| = 2.5 × |Dps| (250% of partial-seizure windows)
    # 
    # NOTE: Paper trained on TUSZ + Siena datasets, OSS only shows TUSZ handling
    # 
    # To add Siena support:
    # 1. Download from PhysioNet (13GB)
    # 2. Parse Seizures-list-PNxx.txt files for annotations
    # 3. Resample from 512Hz to 256Hz
    # 4. Combine with TUSZ in training
    
    # Training loop...
```

---

## 📊 Correct Data Flow

```mermaid
graph TD
    A[TUSZ EDF Files<br/>Unipolar Montage] --> B[Epilepsy2BIDS Eeg Object]
    B --> C[Check Montage<br/>Assert Unipolar]
    C --> D[Channel Handling<br/>19 channels required; no name remap in OSS]
    D --> E[Z-score Normalize<br/>(per-channel over full recording)]
    E --> F[Resample to 256Hz]
    F --> G[Bandpass 0.5-120Hz]
    G --> H[Notch 1Hz, 60Hz]
    H --> I[Sliding Windows<br/>60s, no overlap]
    I --> J[SeizureTransformer<br/>FROM src/]
    J --> K[Per-timestep Probs]
    K --> L[Threshold 0.8]
    L --> M[Morphological Ops]
    M --> N[Remove <2s Events]
    N --> O[Metrics (proposed)]
```

---

## 🎯 Key Differences from Current

### 1. **NO REFERENCE IMPORTS**
- ❌ WRONG: `from wu_2025.architecture import SeizureTransformer`
- ✅ RIGHT: `from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer`

### 2. **COMPLETE PREPROCESSING**
- Z-score normalization FIRST (per-channel, not global)
- Resample to 256Hz if needed
- Bandpass filter (0.5-120Hz, order=3, using lfilter for causal filtering)
- Notch filters (1Hz, 60Hz - hardcoded, no 50Hz option)

### 3. **MONTAGE VALIDATION**
- MUST check for unipolar/referential montage
- Reject bipolar montages or convert them

### 3.1. **Data Requirements (OSS exact)**
- Montage: Unipolar only. The OSS code asserts `Eeg.Montage.UNIPOLAR` and aborts otherwise.
- Channels: Exactly 19 channels required by the model input assertion; OSS does not enforce specific channel names, only shape `(19, 15360)`.
- Sampling rate: Any input fs is resampled to 256 Hz internally; notch filters are computed at 256 Hz.
- Missing channels: No handling in OSS; upstream loader must deliver 19-channel unipolar data.

### 4. **EXACT OSS POST-PROCESSING**
- Threshold: 0.8 (hardcoded in utils.py:101)
- Morphological kernel sizes: 5 (both opening and closing)
- Min event duration: 2.0 seconds
- All parameters are HARDCODED - not configurable

### 5. **CLINICAL METRICS** (Proposed — not in OSS repo)
```python
# src/brain_go_brrr/infra/eval/nedc_metrics.py

class NEDCEvaluator:
    """NEDC-compliant seizure detection metrics."""
    
    def calculate_metrics(self, y_true, y_pred, patient_hours):
        # AUROC
        auroc = roc_auc_score(y_true, y_pred)
        
        # False alarms per 24 hours (event-based recommended)
        # NOTE: Implement FA/24h on merged event segments after post-processing,
        # not per-sample counts. Placeholder below for structure only.
        fa_per_24h = self._events_per_24h(y_true, y_pred, patient_hours)
        
        # TAES (Time-Aligned Event Scoring)
        taes_score = self._calculate_taes(y_true, y_pred)
        
        # Sensitivity at clinical threshold
        sensitivity = self._sensitivity_at_fa_threshold(y_true, y_pred, max_fa=5)
        
        return {
            'auroc': auroc,
            'fa_per_24h': fa_per_24h,
            'taes': taes_score,
            'sensitivity_at_5fa': sensitivity,
        }
```

---

## 🔧 Implementation Steps

1. **COPY** `architecture.py` → `src/brain_go_brrr/infra/ml_models/seizure_transformer.py`
2. **CREATE** `seizure_transformer_utils.py` with preprocessing pipeline
3. **CREATE** `seizure_postprocessor.py` with exact OSS post-processing
4. **UPDATE** `tusz_detection_dataset.py` to use preprocessor
5. **CREATE** `nedc_metrics.py` for clinical evaluation
6. **UPDATE** `train_tusz.py` to import from src/
7. **DELETE** wu_2025 package installation

---

## 🧠 Model Architecture Details (OSS exact)

- Input: `(batch, 19, 15360)`; forward asserts exact shape.
- Encoder: 1D Conv + MaxPool x len(filters)
  - Filters: `[32, 64, 128, 256, 512]`
  - Kernel sizes: `[11, 9, 7, 7, 5, 5, 3]` (applied per stage; see code mapping)
  - Activation: `ELU` in encoder blocks
  - Pre-pool padding: if odd length, right-pad with `-1e10` before `MaxPool1d(2)`
- ResCNN stack: kernels `[3, 3, 3, 3, 2, 3, 2]`
  - Blocks use `BatchNorm1d(eps=1e-3)` + `ReLU` + `SpatialDropout1d(drop_rate)` + Conv
  - For `ker == 2`, manual right padding is applied to emulate TF padding
- Transformer encoder:
  - `d_model=512`, `nhead=4`, `dim_feedforward=2048`, `num_layers=8`
  - Positional encoding: sinusoidal, `dropout=0.1`, `max_len=6000`
- Decoder: Nearest-neighbor upsample x stages with skip additions from encoder
- Output: `Conv1d(in=self.filters[0], out=1, kernel_size=11, padding=5)` + `sigmoid`
- Dropout: `drop_rate=0.1` used in ResCNN `SpatialDropout1d`; positional dropout is fixed at 0.1
- Init: No custom initialization beyond PyTorch defaults
- Skips: Added (not concatenated)
- Note: `forward()` has unused `logits=True` parameter (dead code in OSS)

Code reference: `wu_2025/src/wu_2025/architecture.py` (all details above verified).

---

## 📦 Dependencies & Weights (from OSS)

- Python: `>=3.10`
- Packages: `numpy>=1.25`, `scipy>=1.14.1`, `torch>=2.0.1`, `epilepsy2bids>=0.0.6`
- Weights: Place `model.pth` in `wu_2025/src/wu_2025/` (loaded by `utils.load_models`)
- **CRITICAL**: When loading pretrained weights, model MUST be initialized with default params:
  ```python
  model = SeizureTransformer()  # NO parameters - uses all defaults
  ```
- Output format: Results saved as TSV via `epilepsy2bids.Annotations.saveTsv()` 

---

---

## 📈 Expected Results

With correct implementation:
- AUROC: 0.876 (matching paper Figure 3)
- FA/24h: ≤5 at 90% sensitivity (Note: Paper used 80% threshold in competition)
- Processing: 3.98 seconds per 1-hour EEG (from paper Table II)
- Convergence: ~20 epochs
- Inference batch size: 512 (main.py:17)
- Training batch size: 256 (from paper)
- GPUs used in paper: 2x NVIDIA L40S 46GB (not in OSS code)

---

## ⚠️ Critical Success Factors

### IMPLEMENT FROM CODE (Primary Source):
1. **Unipolar montage** - Model REQUIRES unipolar (will assert error if bipolar)
2. **Exact preprocessing** - Order matters: normalize → resample → bandpass → notch
3. **Window alignment** - 60s windows (15360 samples), 0% overlap for inference
4. **Post-processing params** - All hardcoded: threshold=0.8, kernel=5, min_duration=2.0s
5. **Channel requirements** - EXACTLY 19 channels, hardcoded in architecture
6. **Sampling rate** - Auto-resamples to 256Hz, not configurable
7. **Filter details** - Butterworth order=3, uses lfilter (causal) not filtfilt
8. **Padding strategy** - Zero-padding at END for windows < 15360 samples
9. **Dropout** - Code uses `model.eval()` which DISABLES dropout at test time (standard practice)

---

## 📝 Paper vs Code Discrepancies (For GitHub Issues)

### PAPER CLAIMS (Not in Code):
1. **Datasets**: Paper used TUSZ + Siena Scalp EEG Database
   - Code only shows TUSZ handling via epilepsy2bids
   - To achieve paper parity: Need to add Siena dataset loader

2. **Dropout at test**: Paper says "dropout=0.1 at test time"
   - Likely a PAPER ERROR (dropout should be off at test)
   - Code correctly uses `model.eval()` 

3. **Channel order**: Paper shows channels but no exact sequence
   - Code doesn't validate channel names, only count

4. **Multi-GPU**: Paper used 2x NVIDIA L40S
   - Code has no multi-GPU implementation

5. **Training overlap**: Paper says 75% overlap
   - Code allows configuration but doesn't show 75% default

### ACTION ITEMS:
- **For full paper parity**: Download Siena Scalp EEG Database
  - Available at: https://physionet.org/content/siena-scalp-eeg/1.0.0/
  - DOI: https://doi.org/10.13026/5d4a-j060
  - Size: 20.3 GB uncompressed (13.0 GB ZIP)
  - Contents: 128 hours from 14 subjects at 512Hz
  - 47 seizures total
  - Format: EDF files with 10-20 system + EKG
  - Download: `wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/`
  - Would need custom loader (not in OSS code)
- **Create GitHub issues** asking about:
  - Dropout at test time (likely paper typo?)
  - Exact channel ordering used
  - Multi-GPU training code
  - Why Siena dataset handling not in OSS?

---

## 🚀 Migration Plan

### Phase 1: Copy Architecture
```bash
cp reference_repos/SeizureTransformer/wu_2025/src/wu_2025/architecture.py \
   src/brain_go_brrr/infra/ml_models/seizure_transformer.py
```

### Phase 2: Remove Package
```bash
uv remove wu_2025
```

### Phase 3: Update Imports
```python
# In ALL files:
# FIND: from wu_2025.architecture import SeizureTransformer
# REPLACE: from brain_go_brrr.infra.ml_models.seizure_transformer import SeizureTransformer
```

### Phase 4: Add Preprocessing
- Implement in dataset or wrapper
- Ensure EXACT order from OSS
- Note: Preprocessing split between get_dataloader and preprocess_clip functions

### Phase 5: Validate
- Train for 1 epoch
- Check AUROC > 0.5 (not random)
- Verify preprocessing with unit tests

---

**THIS IS HOW IT SHOULD BE DONE CORRECTLY**
