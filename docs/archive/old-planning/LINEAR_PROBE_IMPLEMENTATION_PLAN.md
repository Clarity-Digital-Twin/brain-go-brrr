# Linear Probe Implementation Plan for EEGPT

## Overview

This document outlines the implementation plan for adding linear probing capabilities to our EEGPT-based system, focusing on abnormality detection and sleep staging as primary use cases.

## Background from EEGPT Paper

### Linear Probing Architecture (Section 2.4)
The EEGPT linear probing method consists of:
1. **Frozen EEGPT Encoder** - Pretrained weights remain fixed
2. **Adaptive Spatial Filter** - 1×1 convolution to align channels between datasets
3. **Linear Classification Head** - Maps features to task-specific logits

Key insight: "This approach helps avoid the overfitting problem when a large parameter model is fine-tuned using a limited number of samples."

### Performance Targets from Paper
- **TUAB Abnormality Detection**: AUROC ≥ 0.93
- **Sleep-EDF Sleep Staging**: Accuracy ≥ 87.5%
- Training time: Minutes on single GPU (vs hours for full fine-tuning)

## Implementation Architecture

```
EEG Input (M channels × T samples)
    ↓
Preprocessing (256 Hz, z-score normalization)
    ↓
Patching (4s windows, 64-sample patches)
    ↓
EEGPT Encoder (frozen)
    ↓
Feature Extraction (summary tokens)
    ↓
Adaptive Spatial Filter (1×1 conv)
    ↓
Linear Head (task-specific)
    ↓
Output (probabilities/classes)
```

## Phase 1: Infrastructure Setup

### 1.1 Re-clone Reference Repositories
```bash
# Clean up empty repos
rm -rf reference_repos/*

# Re-clone key repositories
git clone https://github.com/BINE022/EEGPT.git reference_repos/EEGPT
git clone https://github.com/SPOClab-ca/BENDR.git reference_repos/BENDR
git clone https://github.com/ycq091044/BIOT.git reference_repos/BIOT
git clone https://github.com/935963004/LaBraM.git reference_repos/LaBraM

# Ensure they're gitignored
echo "reference_repos/" >> .gitignore
```

### 1.2 Extract Key Implementation Details
From EEGPT downstream code, we need:
- `downstream/linear_probe_EEGPT_*.py` - Linear probe implementations
- `downstream/Modules/models/EEGPT_mcae_finetune.py` - Model architecture
- `downstream/Data_process/` - Data preprocessing pipelines

## Phase 2: Core Components

### 2.1 Linear Probe Module
```python
# src/brain_go_brrr/models/linear_probe.py

class EEGPTLinearProbe(nn.Module):
    """Linear probe for EEGPT frozen features.

    Architecture from paper:
    - Adaptive spatial filter (1×1 conv)
    - Linear classification head
    - Frozen EEGPT backbone
    """

    def __init__(
        self,
        backbone: EEGPTWrapper,
        n_channels: int,
        n_classes: int,
        embed_dim: int = 512,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Adaptive spatial filter (1×1 conv)
        self.spatial_filter = nn.Conv1d(
            in_channels=n_channels,
            out_channels=58,  # EEGPT expects 58 channels
            kernel_size=1,
            bias=False
        )

        # Classification head
        # EEGPT outputs 4 summary tokens × embed_dim
        self.classifier = nn.Linear(4 * embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, time]

        # Apply spatial filter
        x = self.spatial_filter(x)

        # Extract features with frozen backbone
        with torch.no_grad() if self.training else nullcontext():
            features = self.backbone.extract_features(x)

        # Classify
        logits = self.classifier(features)
        return logits
```

### 2.2 Task-Specific Heads

#### Abnormality Detection Head
```python
class AbnormalityDetectionHead(EEGPTLinearProbe):
    """Binary classification for normal/abnormal EEG."""

    def __init__(self, backbone: EEGPTWrapper, n_channels: int = 23):
        super().__init__(
            backbone=backbone,
            n_channels=n_channels,
            n_classes=2,  # normal/abnormal
            freeze_backbone=True
        )
```

#### Sleep Staging Head
```python
class SleepStagingHead(EEGPTLinearProbe):
    """5-class sleep stage classification."""

    def __init__(self, backbone: EEGPTWrapper, n_channels: int = 2):
        super().__init__(
            backbone=backbone,
            n_channels=n_channels,
            n_classes=5,  # W, N1, N2, N3, REM
            freeze_backbone=True
        )

        # Additional transformer for 30s context (paper detail)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            num_layers=4
        )
```

## Phase 3: Training Infrastructure

### 3.1 Lightning Module
```python
# src/brain_go_brrr/training/linear_probe_trainer.py

class LinearProbeTrainer(pl.LightningModule):
    """PyTorch Lightning module for linear probe training."""

    def __init__(
        self,
        probe_model: EEGPTLinearProbe,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01
    ):
        super().__init__()
        self.model = probe_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        # Metrics
        acc = (logits.argmax(1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def configure_optimizers(self):
        # Only optimize probe parameters
        probe_params = [
            p for p in self.model.parameters()
            if p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            probe_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            total_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
```

### 3.2 Data Module
```python
# src/brain_go_brrr/data/probe_datasets.py

class TUABDataModule(pl.LightningDataModule):
    """TUAB abnormality detection dataset."""

    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 64,
        num_workers: int = 4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        # Follow EEGPT paper split: 70/15/15
        self.train_dataset = TUABDataset(
            self.data_dir / "train",
            transform=self.get_transform()
        )
        self.val_dataset = TUABDataset(
            self.data_dir / "val",
            transform=self.get_transform()
        )
        self.test_dataset = TUABDataset(
            self.data_dir / "test",
            transform=self.get_transform()
        )

    def get_transform(self):
        return Compose([
            Resample(256),  # Ensure 256 Hz
            Segment(duration=30.0, stride=30.0),  # 30s windows
            ZScoreNormalize(),
            ToTensor()
        ])
```

## Phase 4: Experiment Scripts

### 4.1 Abnormality Detection Experiment
```python
# experiments/abnormality_probe/run.py

import hydra
from omegaconf import DictConfig
from pathlib import Path

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Load pretrained EEGPT
    backbone = EEGPTWrapper.load_from_checkpoint(
        cfg.model.checkpoint_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create probe model
    probe = AbnormalityDetectionHead(
        backbone=backbone,
        n_channels=cfg.data.n_channels
    )

    # Data module
    datamodule = TUABDataModule(
        data_dir=Path(cfg.data.root_dir),
        batch_size=cfg.training.batch_size
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=16,  # Mixed precision as in paper
        logger=WandbLogger(project="eegpt-probes"),
        callbacks=[
            ModelCheckpoint(
                monitor="val_auroc",
                mode="max",
                save_top_k=1
            ),
            EarlyStopping(
                monitor="val_auroc",
                patience=10,
                mode="max"
            )
        ]
    )

    # Train
    trainer.fit(probe, datamodule)

    # Test
    results = trainer.test(probe, datamodule)
    print(f"Test AUROC: {results[0]['test_auroc']:.4f}")
```

### 4.2 Configuration
```yaml
# experiments/abnormality_probe/config.yaml

model:
  checkpoint_path: "data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt"
  embed_dim: 512

data:
  root_dir: "data/datasets/external/tuh_eeg_abnormal/v3.0.0"
  n_channels: 23  # TUAB uses 23 channels

training:
  batch_size: 64
  epochs: 10
  learning_rate: 1e-3
  weight_decay: 0.01

seed: 42
```

## Phase 5: Integration & Testing

### 5.1 Unit Tests
```python
# tests/models/test_linear_probe.py

def test_linear_probe_frozen_backbone():
    """Test that backbone remains frozen during training."""
    backbone = create_mock_eegpt()
    probe = EEGPTLinearProbe(backbone, n_channels=23, n_classes=2)

    # Check backbone is frozen
    for param in probe.backbone.parameters():
        assert not param.requires_grad

    # Check probe parameters are trainable
    probe_params = [p for p in probe.parameters() if p.requires_grad]
    assert len(probe_params) > 0
```

### 5.2 Integration Tests
```python
# tests/integration/test_probe_training.py

@pytest.mark.slow
def test_abnormality_probe_training():
    """Test full training pipeline reaches target metrics."""
    # Use small subset of TUAB
    probe, datamodule = setup_test_probe()

    trainer = pl.Trainer(
        max_epochs=2,
        fast_dev_run=False,
        accelerator="cpu"
    )

    trainer.fit(probe, datamodule)
    results = trainer.test(probe, datamodule)

    # Should reach reasonable AUROC even on small data
    assert results[0]["test_auroc"] > 0.7
```

## Phase 6: API Integration

### 6.1 FastAPI Endpoint
```python
# src/brain_go_brrr/api/endpoints/analysis.py

@router.post("/analyze/abnormality")
async def analyze_abnormality(
    file: UploadFile,
    background_tasks: BackgroundTasks
) -> AnalysisResponse:
    """Analyze EEG for abnormalities using EEGPT linear probe."""

    # Load model
    probe = load_abnormality_probe()

    # Process file
    eeg_data = await load_eeg_file(file)

    # Run inference
    with torch.no_grad():
        probabilities = probe(eeg_data)

    # Generate report
    report = generate_abnormality_report(probabilities)

    return AnalysisResponse(
        task="abnormality_detection",
        results=report,
        model_version="eegpt-probe-v1"
    )
```

## Timeline & Milestones

### Week 1: Infrastructure & Setup
- [ ] Re-clone and study reference implementations
- [ ] Implement core LinearProbe module
- [ ] Set up training infrastructure

### Week 2: Abnormality Detection
- [ ] Implement TUAB data module
- [ ] Train abnormality detection probe
- [ ] Achieve AUROC > 0.90 on validation set

### Week 3: Sleep Staging
- [ ] Implement Sleep-EDF data module
- [ ] Train sleep staging probe with context encoder
- [ ] Achieve accuracy > 85% on validation set

### Week 4: Integration & Polish
- [ ] Add API endpoints for both tasks
- [ ] Create visualization and reporting tools
- [ ] Document model cards and usage

## Success Criteria

1. **Abnormality Detection**
   - AUROC ≥ 0.90 on TUAB test set
   - Training time < 1 hour on single GPU
   - Inference speed > 30× real-time

2. **Sleep Staging**
   - Accuracy ≥ 85% on Sleep-EDF test set
   - Proper handling of 30s context windows
   - Hypnogram visualization

3. **Code Quality**
   - 100% type hints and docstrings
   - Passes all linting and tests
   - Reproducible results with fixed seeds

## Next Steps

1. **Immediate**: Re-clone EEGPT repository and study implementation
2. **Today**: Create feature branch and implement LinearProbe base class
3. **This Week**: Get abnormality detection probe training successfully

## References

- EEGPT Paper: https://arxiv.org/abs/2410.20150
- EEGPT Code: https://github.com/BINE022/EEGPT
- TUAB Dataset: https://www.isip.piconepress.com/projects/tuh_eeg/
- Sleep-EDF: https://physionet.org/content/sleep-edfx/
