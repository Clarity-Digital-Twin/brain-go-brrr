#!/usr/bin/env python3
"""Train SeizureTransformer on TUSZ dataset for temporal seizure detection.

This replicates the training setup from Wu et al. 2025:
- 60s windows at 256Hz
- Z-score → resample → bandpass (0.5-120Hz) → notch (1Hz, 60Hz)
- Post-processing: threshold 0.8 → morphological ops → remove < 2s
- Target: AUROC 0.876 on TUSZ test set
"""

import os
import random
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
from brain_go_brrr.infra.ml_models.seizure_transformer_utils import SeizurePreprocessor


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
        y = y.to(device).float()  # (B, 15360) per-timestep labels

        # Forward pass - model expects (B, C, T)
        logits = model(x)  # (B, 15360) per-timestep predictions

        # Binary cross-entropy loss with per-timestep labels
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

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
            y = y.to(device).float()  # Now scalar labels (B,)

            # Forward pass
            logits = model(x)  # (B, 15360)
            probs = torch.sigmoid(logits)

            # Compute loss - expand scalar labels to match timesteps
            y_expanded = y.unsqueeze(1).expand(-1, logits.shape[1])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y_expanded)
            total_loss += loss.item()

            # Store window-level predictions for AUROC
            window_probs = probs.mean(dim=1)  # Average across timesteps
            all_preds.append(window_probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Compute AUROC
    from sklearn.metrics import roc_auc_score

    all_preds = np.concatenate(all_preds)  # Already window-level
    all_labels = np.concatenate(all_labels)

    # Handle case where only one class is present
    if len(np.unique(all_labels)) == 1:
        print(f"Warning: Only one class in validation set (all {all_labels[0]}s)")
        auroc = 0.5  # Default AUROC for single class
    else:
        auroc = roc_auc_score(all_labels, all_preds)

    return total_loss / len(dataloader), auroc


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Set seeds for reproducibility
    set_seeds(42)

    # Configuration
    data_root = os.environ.get(
        "BGB_DATA_ROOT", "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"
    )
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

    # Create SSOT preprocessor
    preprocessor = SeizurePreprocessor(target_fs=256)

    # Create datasets with memory-efficient loading
    print("Loading TUSZ training set (memory-efficient mode)...")
    train_ds = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="train",
        cfg=cfg,
        max_windows=1000,  # Start with 1k windows for quick testing
        preprocessor=preprocessor,  # Use SSOT preprocessing
        ensure_unipolar=False,  # TUSZ has mixed montages
        return_timestep_labels=True,  # Use per-timestep labels for training
    )

    print("Loading TUSZ dev set...")
    val_ds = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="dev",
        cfg=WindowConfig(
            fs=256,
            window_sec=60.0,
            stride_sec=30.0,  # Some overlap to get more windows
            positive_fraction=0.2,  # Same as training
        ),
        max_windows=500,  # Smaller validation set for quick testing
        preprocessor=preprocessor,  # Use SSOT preprocessing
        ensure_unipolar=False,  # TUSZ has mixed montages
        return_timestep_labels=False,  # Use scalar labels for validation AUROC
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_ds,
        batch_size=8,  # Reduced batch size for 41M param model
        shuffle=True,
        num_workers=2,  # Reduced workers to save memory
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,  # Keep workers alive between epochs
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=16,  # Reduced batch size
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use SeizureTransformerWrapper for correct Wu 2025 architecture
    print("Loading SeizureTransformer Wu 2025 architecture...")
    from brain_go_brrr.infra.ml_models.seizure_transformer_wu2025 import SeizureTransformer

    # Create the actual model (Wu 2025 architecture with CNN+Transformer)
    model = SeizureTransformer(
        in_channels=19,
        in_samples=15360,  # 60s @ 256Hz
        drop_rate=0.1,
    )

    # Or use through wrapper for preprocessing/postprocessing
    # wrapper = SeizureTransformerWrapper(
    #     model=model,
    #     n_channels=19,
    #     fs=256,
    #     window_samples=15360,
    # )

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
