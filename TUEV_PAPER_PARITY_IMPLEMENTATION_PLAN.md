# 📋 TUEV PAPER PARITY IMPLEMENTATION PLAN

**Created**: September 9, 2025  
**Purpose**: SINGLE SOURCE OF TRUTH for achieving EEGPT paper's 62.32% BAC on TUEV  
**Current Status**: Cache 80% built with WRONG approach (20-ch preprocessing)  
**Decision Required**: Kill cache NOW and implement paper parity

---

## 🎯 WHAT THE PAPER ACTUALLY DID (VERIFIED FROM REFERENCE CODE)

### Architecture (from `reference_repos/EEGPT/downstream_tueg/`)
1. **Input**: 23 channels from TUEV (keeps ALL channels including A1, A2, T1, T2)
2. **Learnable Mapper**: Conv2dWithConstraint(23, 20, kernel_size=1)
   - Followed by: BatchNorm2d(20) → GELU() → Dropout(0.8)
   - Then: Conv2d(20, 20, kernel_size=(1,55), groups=20)
3. **Always Enabled**: `use_chan_conv=True` for TUEV
4. **NO Preprocessing**: No channel dropping, no Fpz synthesis

### Hyperparameters (from reference implementation)
- `lr = 5e-4`
- `weight_decay = 0.05`
- `label_smoothing = 0.1` (via LabelSmoothingCrossEntropy)
- `batch_size = 64` (paper text says 500 but code uses 64)
- **NO class weights** (plain CrossEntropyLoss with smoothing)
- **NO balanced sampling** (standard random sampling)

### Target Metrics (from EEGPT Table 3)
- **Balanced Accuracy (BAC)**: 62.32% ± 1.14%
- **Weighted F1**: 81.87% ± 0.63% (misleading due to 99.5% class imbalance)
- **Cohen's Kappa**: 0.635 ± 0.013

---

## ❌ WHAT WE'RE DOING WRONG

### Current Implementation
1. **Preprocessing to 20 channels**: Dropping A1/A2 in `tuev_preprocessor.py`
2. **Fpz synthesis**: Interpolating as (Fp1+Fp2)/2 or zeros
3. **No learnable mapper**: Missing the Conv2d(23→20) module entirely
4. **Result**: ~50-55% BAC expected (not 62%)

### Why This Matters
- The mapper is NOT optional - it's THE architecture that achieved 62% BAC
- Our preprocessing approach fundamentally differs from the paper
- We're testing a different hypothesis, not reproducing their results

---

## 🔴 SENIOR AUDITOR VERIFICATION - ALL ISSUES FIXED

### Issues Found and Fixed:
1. ✅ **Conv2dWithConstraint**: Correctly identified as missing, needs to be added
2. ✅ **MNE API**: Fixed to use `raw.pick()` not `raw.copy().pick_channels()`
3. ✅ **Channel aliasing**: Added warning about T3→T7 mapping, using canonical names
4. ✅ **Cache paths**: Fixed to use `$BGB_DATA_ROOT/cache/tuev_mne_fixed` not hardcoded
5. ✅ **Missing imports**: Added `import os` to test files
6. ✅ **23-channel list**: Using T7/T8/P7/P8 (canonical) not T3/T4/T5/T6 (legacy)
7. ✅ **Training integration**: Properly pass channel_mapper to train_epoch function

## ⚠️ INTEGRATION NOTES - AVOID REDUNDANCY

### What We Already Have (DO NOT RECREATE):
1. **Constraint layers**: `src/brain_go_brrr/domain/constraints.py` has `LinearWithConstraint` and `Conv1dWithConstraint`
   - We just need to ADD `Conv2dWithConstraint` to this file
2. **Probe architecture**: `src/brain_go_brrr/infra/ml_models/linear_probe.py` has `TwoLayerProbe`
   - Training already uses this, don't create a new one
3. **EEGPT wrapper**: `src/brain_go_brrr/infra/ml_models/eegpt_wrapper.py` 
   - Training already uses this, don't modify
4. **Dataset base**: `src/brain_go_brrr/infra/data/tuev_dataset.py` exists
   - Just add `use_paper_parity` parameter, don't rewrite
5. **Preprocessor base**: `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py` exists
   - Just add 23-channel mode, don't rewrite

### What We're Adding (NEW):
1. **Conv2dWithConstraint**: One new class in existing `constraints.py`
2. **TUEVChannelMapper**: New module in `infra/ml_models/`
3. **use_paper_parity flag**: Parameter in existing dataset/preprocessor
4. **channel_mapper integration**: Small changes to training script

## ✅ IMPLEMENTATION PLAN

### IMMEDIATE DECISION: Kill Current Cache Build
```bash
# 1. Find and kill the cache build process
tmux list-sessions
tmux attach -t tuev_cache
# Press Ctrl+C to stop

# 2. Clean up partial cache (use actual cache path)
export BGB_DATA_ROOT="${BGB_DATA_ROOT:-/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data}"
rm -rf "$BGB_DATA_ROOT/cache/tuev_mne_fixed/"
```

### Phase 1: Create Channel Mapper Module (30 min)

#### 1.1 Add Conv2dWithConstraint to `src/brain_go_brrr/domain/constraints.py`
```python
# Add this class to the existing constraints.py file:

class Conv2dWithConstraint(nn.Conv2d):
    """2D Convolution with weight norm constraint.
    
    Matches EEGPT reference implementation for channel mapping.
    """
    
    def __init__(
        self, *args: Any, do_weight_norm: bool = True, max_norm: float = 1.0, **kwargs: Any
    ) -> None:
        """Initialize constrained conv2d layer."""
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight constraint."""
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)
```

#### 1.2 Create `src/brain_go_brrr/infra/ml_models/channel_mapper.py`
```python
import torch
import torch.nn as nn
from brain_go_brrr.domain.constraints import Conv2dWithConstraint

class TUEVChannelMapper(nn.Module):
    """
    Learnable 23→20 channel mapper for TUEV paper parity.
    Matches EEGPT reference implementation exactly.
    """
    
    def __init__(self, in_channels: int = 23, out_channels: int = 20, dropout: float = 0.8):
        super().__init__()
        
        # Spatial convolution (23→20 learned mapping)
        self.spatial_conv = nn.Sequential(
            Conv2dWithConstraint(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # Temporal convolution (depthwise)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 55), 
                     groups=out_channels, padding=(0, 27), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 23, T) or (B, 23, H, T) tensor
        Returns:
            (B, 20, T) or (B, 20, H, T) tensor
        """
        # Add height dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(2)  # (B, C, T) -> (B, C, 1, T)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply mappings
        x = self.spatial_conv(x)   # (B, 23, 1, T) -> (B, 20, 1, T)
        x = self.temporal_conv(x)  # (B, 20, 1, T) -> (B, 20, 1, T)
        
        # Remove height dimension if we added it
        if squeeze_output:
            x = x.squeeze(2)  # (B, 20, 1, T) -> (B, 20, T)
        
        return x
```

#### 1.2 Create tests `tests/unit/infra/ml_models/test_channel_mapper.py`
```python
import pytest
import torch
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper

def test_channel_mapper_shapes():
    """Test that mapper correctly transforms 23→20 channels."""
    mapper = TUEVChannelMapper()
    
    # Test 3D input
    x_3d = torch.randn(32, 23, 1024)
    y_3d = mapper(x_3d)
    assert y_3d.shape == (32, 20, 1024)
    
    # Test 4D input
    x_4d = torch.randn(32, 23, 1, 1024)
    y_4d = mapper(x_4d)
    assert y_4d.shape == (32, 20, 1, 1024)

def test_gradient_flow():
    """Test that gradients flow through the mapper."""
    mapper = TUEVChannelMapper()
    x = torch.randn(1, 23, 256, requires_grad=True)
    y = mapper(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None
    assert torch.any(x.grad != 0)

def test_deterministic_init():
    """Test reproducible initialization with seed."""
    torch.manual_seed(42)
    mapper1 = TUEVChannelMapper()
    
    torch.manual_seed(42)
    mapper2 = TUEVChannelMapper()
    
    # Check weights are identical
    for p1, p2 in zip(mapper1.parameters(), mapper2.parameters()):
        assert torch.allclose(p1, p2)
```

### Phase 2: Modify Dataset to Keep 23 Channels (45 min)

#### 2.1 Update `src/brain_go_brrr/infra/preprocessing/tuev_preprocessor.py`
```python
# Add to imports:
from typing import List, Optional

# Add these constants after existing imports:
# CRITICAL: Use CANONICALIZED names (T7/T8/P7/P8) not legacy (T3/T4/T5/T6)
# Or bypass canonicalization in paper parity mode
CHANNELS_TUEV_23_CANONICAL = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',  # Modern names
    'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
]

# If keeping legacy names, bypass canonicalization:
CHANNELS_TUEV_23_LEGACY = [
    'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
    'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',  # Legacy names
    'A1', 'A2', 'FZ', 'CZ', 'PZ', 'T1', 'T2'
]

# Modify the __init__ method:
class TUEVPreprocessor(TUABPreprocessor):
    def __init__(self, use_paper_parity: bool = False):  # Default to False for compatibility
        """
        Args:
            use_paper_parity: If True, keep all 23 channels for learned mapper.
                            If False, use existing 20-ch preprocessing approach.
        """
        super().__init__()
        self.use_paper_parity = use_paper_parity
        
        if use_paper_parity:
            # Override parent settings for 23-channel mode
            # DECISION: Use canonical names after aliasing, or bypass aliasing
            self.STANDARD_CHANNELS = CHANNELS_TUEV_23_CANONICAL  # Use modern T7/T8/P7/P8
            self.n_channels = 23
            self.bypass_canonicalization = False  # Keep canonicalization
            logger.info("TUEVPreprocessor: Paper parity mode - keeping 23 channels")
        else:
            # Use existing 20-channel mapping
            self.STANDARD_CHANNELS = CHANNELS_TUEV_20
            self.n_channels = 20
            self.bypass_canonicalization = False
            logger.info("TUEVPreprocessor: Standard mode - mapping to 20 channels")
    
    def _apply_channel_mapping(self, raw):
        """Override parent method for paper parity mode."""
        if self.use_paper_parity:
            # Keep all 23 channels, no synthesis
            return self._apply_23_channel_mapping(raw)
        else:
            # Use existing 20-channel mapping with Fpz synthesis
            return super()._apply_channel_mapping(raw)
    
    def _apply_23_channel_mapping(self, raw):
        """Keep all 23 TUEV channels without synthesis."""
        # Check which channels are available
        available = [ch for ch in self.STANDARD_CHANNELS if ch in raw.ch_names]
        missing = [ch for ch in self.STANDARD_CHANNELS if ch not in raw.ch_names]
        
        if missing:
            logger.warning(f"Missing channels for 23-ch parity: {missing}")
            # For paper parity, we should have all 23 channels
            # If not, the dataset may be incomplete
        
        # Use raw.pick() (modern MNE API) not pick_channels
        raw_copy = raw.copy()
        raw_copy.pick(available, ordered=True)
        
        # Ensure we have exactly 23 channels for mapper
        if len(raw_copy.ch_names) != 23:
            raise ValueError(
                f"Paper parity requires exactly 23 channels, got {len(raw_copy.ch_names)}"
            )
        
        return raw_copy
```

#### 2.2 Update `src/brain_go_brrr/infra/data/tuev_dataset.py`
```python
# Modify the __init__ method to add use_paper_parity parameter:
class TUEVMNEDataset:
    def __init__(
        self,
        root_dir: Path,
        split: str = "train",
        cache_dir: Optional[Path] = None,
        force_rebuild: bool = False,
        use_paper_parity: bool = False,  # Add this parameter
        **kwargs
    ):
        """
        Args:
            use_paper_parity: If True, use 23 channels for paper parity.
                            If False, use existing 20-channel approach.
        """
        self.use_paper_parity = use_paper_parity
        
        # Modify cache directory based on mode
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'brain_go_brrr'
        
        if use_paper_parity:
            self.cache_dir = cache_dir / "tuev_23ch_paper_parity" / split
        else:
            self.cache_dir = cache_dir / "tuev_mne" / split  # Keep existing path
        
        # Update preprocessor initialization
        self.preprocessor = TUEVPreprocessor(use_paper_parity=use_paper_parity)
        
        # Update expected channels
        self.n_channels = 23 if use_paper_parity else 20
        
        # Rest of initialization...
        super().__init__(root_dir, split, self.cache_dir, force_rebuild, **kwargs)
```

### Phase 3: Integrate Mapper in Training (30 min)

#### 3.1 Modify `experiments/eegpt_linear_probe/train_tuev_mne.py`
```python
# Add to imports (around line 38):
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper

# After creating the probe (around line 475), add mapper initialization:
# Create channel mapper if using paper parity
use_channel_mapper = config.get('model', {}).get('use_channel_mapper', False)
if use_channel_mapper:
    logger.info("Initializing 23→20 channel mapper for paper parity")
    channel_mapper = TUEVChannelMapper(
        in_channels=23,
        out_channels=20,
        dropout=config.get('model', {}).get('mapper_dropout', 0.8)
    ).to(device)
else:
    channel_mapper = None
    logger.info("No channel mapper - using preprocessed 20 channels")

# Modify optimizer creation (around line 478) to include mapper params:
if channel_mapper is not None:
    # Include both probe and mapper parameters
    optimizer = torch.optim.AdamW([
        {'params': probe.parameters()},
        {'params': channel_mapper.parameters()}
    ], lr=config['training']['learning_rate'], 
       weight_decay=config['training']['weight_decay'])
else:
    # Only probe parameters (existing code)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

# Pass channel_mapper as parameter to train_epoch
def train_epoch(
    model,
    probe,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    output_dir=None,
    global_step=0,
    config=None,
    epoch_indices=None,
    start_batch=0,
    channel_mapper=None,  # ADD THIS PARAMETER
):
    # In the training loop (around line 159):
    # Apply channel mapper if provided
    if channel_mapper is not None:
        x = channel_mapper(x)  # (B, 23, T) -> (B, 20, T)
    
    # Then continue with existing feature extraction:
    with torch.no_grad():
        features = model.extract_features(x, summary=False)

# UPDATE THE TRAINING LOOP CALL (around line 600+):
# When calling train_epoch, pass the channel_mapper:
global_step = train_epoch(
    model=model,
    probe=probe,
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device=device,
    epoch=epoch,
    output_dir=output_dir,
    global_step=global_step,
    config=config,
    epoch_indices=epoch_indices,
    start_batch=start_batch,
    channel_mapper=channel_mapper  # ADD THIS
)

# Similarly update evaluate() function signature if needed
```

#### 3.2 Update config `experiments/eegpt_linear_probe/configs/tuev_paper_parity.yaml`
```yaml
# New config for paper parity
data:
  dataset: "tuev"
  root_dir: "${BGB_DATA_ROOT}/datasets/tuev"
  cache_dir: "${BGB_CACHE_DIR}/tuev_23ch_paper_parity"  # NEW cache
  
  window_samples: 1024
  sampling_rate: 256
  n_channels: 23  # Keep ALL channels
  n_classes: 6
  
  batch_size: 64
  num_workers: 4
  pin_memory: true

model:
  eegpt_checkpoint: "${BGB_DATA_ROOT}/models/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
  freeze_backbone: true
  
  # Paper parity settings
  use_channel_mapper: true
  mapper_dropout: 0.8  # From EEGPT reference

training:
  n_epochs: 100
  learning_rate: 5.0e-4
  weight_decay: 0.05
  label_smoothing: 0.1
  weighted_loss: false  # NO class weights per paper
  
  gradient_clip: 1.0
  patience: 20
  min_delta: 0.001

validation:
  n_runs: 3
  seeds: [42, 123, 456]
  
  targets:
    balanced_accuracy: 0.6232
    weighted_f1: 0.8187
    cohen_kappa: 0.6351
```

### Phase 4: Rebuild Cache with 23 Channels (4-6 hours)

#### 4.1 Create cache builder script `experiments/eegpt_linear_probe/scripts/build_tuev_23ch_cache.sh`
```bash
#!/bin/bash
set -e

# Set environment variables (adjust paths as needed)
export BGB_DATA_ROOT="${BGB_DATA_ROOT:-/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data}"
export BGB_CACHE_DIR="${BGB_CACHE_DIR:-$BGB_DATA_ROOT/cache}"

echo "Building TUEV 23-channel cache for paper parity..."
echo "Data root: $BGB_DATA_ROOT"
echo "Cache dir: $BGB_CACHE_DIR"
echo "This will take 4-6 hours. Run in tmux!"

# Build cache for both train and eval splits
for SPLIT in train eval; do
    echo "Building $SPLIT split..."
    python -c "
import sys
sys.path.insert(0, '.')  # Ensure we can import brain_go_brrr
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
from pathlib import Path

dataset = TUEVMNEDataset(
    root_dir=Path('$BGB_DATA_ROOT/datasets/tuev'),
    split='$SPLIT',
    cache_dir=Path('$BGB_CACHE_DIR'),
    force_rebuild=True,
    use_paper_parity=True  # 23 channels
)

print(f'Built {len(dataset)} windows for $SPLIT split')
print(f'Channels: {dataset.n_channels}')
print(f'Cache location: {dataset.cache_dir}')
"
done

echo "Cache build complete!"
```

#### 4.2 Launch cache build in tmux
```bash
tmux new -s tuev_23ch_cache
cd experiments/eegpt_linear_probe/scripts
./build_tuev_23ch_cache.sh
# Detach with Ctrl+B, D
```

### Phase 5: Training with Paper Parity (8-12 hours)

#### 5.1 Create launch script `experiments/eegpt_linear_probe/scripts/launch_tuev_paper_parity.sh`
```bash
#!/bin/bash
set -e

export BGB_DATA_ROOT=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data
export BGB_CACHE_DIR=/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data/cache

cd experiments/eegpt_linear_probe

echo "Starting TUEV training with paper parity (23ch + mapper)..."
python train_tuev_mne.py \
    --config configs/tuev_paper_parity.yaml \
    --run_name tuev_paper_parity_$(date +%Y%m%d_%H%M%S) \
    --cache_dir $BGB_CACHE_DIR/tuev_23ch_paper_parity
```

#### 5.2 Launch training
```bash
tmux new -s tuev_parity_training
cd experiments/eegpt_linear_probe/scripts
./launch_tuev_paper_parity.sh
# Detach with Ctrl+B, D
```

### Phase 6: Verification & Testing (1 hour)

#### 6.1 Integration test `tests/integration/test_tuev_paper_parity.py`
```python
import pytest
import torch
from pathlib import Path
from brain_go_brrr.infra.data.tuev_dataset import TUEVMNEDataset
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
from brain_go_brrr.infra.ml_models.eegpt_wrapper import create_normalized_eegpt

@pytest.mark.integration
def test_paper_parity_pipeline():
    """Test complete pipeline with 23ch input + mapper."""
    import os  # Add missing import
    from pathlib import Path
    
    # Load one sample with 23 channels
    dataset = TUEVMNEDataset(
        root_dir=Path(os.environ['BGB_DATA_ROOT']) / 'datasets/tuev',
        split='eval',
        use_paper_parity=True
    )
    
    x, y = dataset[0]
    assert x.shape[0] == 23, f"Expected 23 channels, got {x.shape[0]}"
    
    # Apply mapper
    mapper = TUEVChannelMapper()
    x_mapped = mapper(torch.from_numpy(x).unsqueeze(0))
    assert x_mapped.shape[1] == 20, f"Expected 20 channels after mapping, got {x_mapped.shape[1]}"
    
    # Feed to EEGPT
    model = create_normalized_eegpt()
    with torch.no_grad():
        features = model.extract_features(x_mapped, summary=False)
    assert features.shape == (1, 16, 4, 512), f"Unexpected feature shape: {features.shape}"

def test_cache_has_23_channels():
    """Verify cache was built with 23 channels."""
    import os  # Add missing import
    from pathlib import Path
    
    cache_dir = Path(os.environ['BGB_CACHE_DIR']) / 'tuev_23ch_paper_parity'
    meta_file = cache_dir / 'train' / 'META.json'
    
    assert meta_file.exists(), "Cache not built yet"
    
    import json
    with open(meta_file) as f:
        meta = json.load(f)
    
    assert meta['n_channels'] == 23, f"Cache has {meta['n_channels']} channels, expected 23"
```

#### 6.2 Run all tests
```bash
# Unit tests for mapper
pytest tests/unit/infra/ml_models/test_channel_mapper.py -xvs

# Integration test (after cache is built)
pytest tests/integration/test_tuev_paper_parity.py -xvs

# Smoke test training
cd experiments/eegpt_linear_probe
python train_tuev_mne.py --config configs/tuev_paper_parity.yaml --debug --max_epochs 1
```

---

## 📊 SUCCESS CRITERIA

### Immediate (Day 1)
- [ ] Channel mapper module created and tested
- [ ] 23-channel cache build started in tmux
- [ ] All unit tests pass

### Short-term (Day 2-3)
- [ ] Cache build complete (~160k windows)
- [ ] Training launched with paper parity config
- [ ] First epoch shows BAC > 0.30 (vs 0.22 currently)

### Target (Week 1)
- [ ] Achieve BAC ≥ 60% on eval set
- [ ] Confirm all 6 classes have non-zero recall
- [ ] Document final hyperparameters that worked

### Stretch Goal
- [ ] Achieve BAC ≥ 62.32% (exact paper match)
- [ ] Run 3 seeds and report mean ± std
- [ ] Ablation: compare with/without mapper

---

## 🚨 CRITICAL REMINDERS

1. **KILL CURRENT CACHE BUILD** - It's using wrong approach
2. **NO PREPROCESSING SYNTHESIS** - Let mapper learn Fpz
3. **NO CLASS WEIGHTS** - Paper doesn't use them
4. **USE EXACT HYPERPARAMS** - lr=5e-4, wd=0.05, smoothing=0.1
5. **MONITOR BAC NOT F1** - F1 is misleading with 99.5% imbalance

---

## 📝 COMMAND SUMMARY

```bash
# 1. Kill wrong cache build
tmux attach -t tuev_cache && Ctrl+C

# 2. Run tests for new mapper
pytest tests/unit/infra/ml_models/test_channel_mapper.py -xvs

# 3. Start correct cache build
tmux new -s tuev_23ch_cache
./build_tuev_23ch_cache.sh

# 4. Monitor cache progress
tmux attach -t tuev_23ch_cache

# 5. Start training (after cache done)
tmux new -s tuev_parity_training
./launch_tuev_paper_parity.sh

# 6. Monitor training
tail -f experiments/eegpt_linear_probe/logs/tuev_paper_parity_*.log | grep -i "bac\|balanced"
```

---

**THIS IS THE ONLY PLAN. FOLLOW IT EXACTLY.**