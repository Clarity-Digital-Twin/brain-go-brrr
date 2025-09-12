#!/usr/bin/env python3
"""Train SeizureTransformer on TUSZ dataset for temporal seizure detection.

This replicates the training setup from Wu et al. 2025:
- 60s windows at 256Hz
- Z-score → resample → bandpass (0.5-120Hz) → notch (1Hz, 60Hz)
- Post-processing: threshold 0.8 → morphological ops → remove < 2s
- Target: AUROC 0.876 on TUSZ test set
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from src - NO REIMPLEMENTATION
from brain_go_brrr.infra.data.tusz_detection_dataset import (
    TUSZDetectionDataset,
    WindowConfig,
)
from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import (
    SeizureTransformerWrapper,
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Training")):
        x = x.to(device)  # (B, 19, 15360)
        y = y.to(device).float()  # (B,) binary labels

        # Forward pass - model expects (B, C, T)
        logits = model(x)  # (B, 15360) per-timestep predictions

        # Create per-timestep labels
        y_expanded = y.unsqueeze(1).expand(-1, logits.shape[1])  # (B, 15360)

        # Binary cross-entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y_expanded)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Validate model and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x = x.to(device)
            y = y.to(device).float()

            # Forward pass
            logits = model(x)
            probs = torch.sigmoid(logits)

            # Compute loss
            y_expanded = y.unsqueeze(1).expand(-1, logits.shape[1])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_expanded)
            total_loss += loss.item()

            # Store predictions for AUROC
            all_preds.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Compute AUROC
    from sklearn.metrics import roc_auc_score

    all_preds = np.concatenate([p.mean(axis=1) for p in all_preds])  # Average per window
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_preds)

    return total_loss / len(dataloader), auroc


def main():
    # Configuration
    data_root = os.environ.get("BGB_DATA_ROOT")
    if not data_root:
        raise ValueError("BGB_DATA_ROOT environment variable not set")

    tusz_root = Path(data_root) / "datasets/tusz/edf"
    if not tusz_root.exists():
        raise ValueError(f"TUSZ dataset not found at {tusz_root}")

    # Training config matching paper
    cfg = WindowConfig(
        fs=256,
        window_sec=60.0,  # 60s windows
        stride_sec=15.0,  # 75% overlap for training
        positive_fraction=0.2,  # Balance positive samples
    )

    # Create datasets
    print("Loading TUSZ training set...")
    train_ds = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="train",
        cfg=cfg,
    )

    print("Loading TUSZ dev set...")
    val_ds = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="dev",
        cfg=WindowConfig(
            fs=256,
            window_sec=60.0,
            stride_sec=60.0,  # No overlap for validation
        ),
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=16,  # Adjust based on GPU memory
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model using wrapper's build function
    print("Building SeizureTransformer model...")
    
    # The wrapper will try to import wu_2025, if not available it needs a build_fn
    def build_seizure_transformer(n_channels: int) -> nn.Module:
        """Build a simple U-Net that outputs per-timestep predictions."""
        # This is a placeholder - the real SeizureTransformer would go here
        # For now, create a simple model that matches the expected interface
        import torch.nn as nn
        
        class SimpleSeizureModel(nn.Module):
            def __init__(self, n_channels: int = 19, window_samples: int = 15360):
                super().__init__()
                self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
                self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
                self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
                self.conv4 = nn.Conv1d(64, 1, kernel_size=1)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # x: (B, C, T)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.conv4(x)  # (B, 1, T)
                return x.squeeze(1)  # (B, T)
        
        return SimpleSeizureModel(n_channels, 15360)
    
    wrapper = SeizureTransformerWrapper(
        build_fn=build_seizure_transformer,
        n_channels=19,
        fs=256,
        window_samples=15360,
    )
    model = wrapper.model

    model = model.to(device)

    # Training setup matching paper
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=1e-3,
        weight_decay=2e-5,
    )

    # Training loop
    best_auroc = 0.0
    patience = 12
    patience_counter = 0

    for epoch in range(100):
        print(f"\nEpoch {epoch + 1}/100")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, auroc = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, AUROC: {auroc:.4f}")

        # Early stopping
        if auroc > best_auroc:
            best_auroc = auroc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"seizure_transformer_best_auroc_{auroc:.3f}.pt")
            print(f"New best AUROC: {auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

    print(f"\nTraining complete! Best AUROC: {best_auroc:.4f}")
    print("Target from paper: 0.876")


if __name__ == "__main__":
    main()
