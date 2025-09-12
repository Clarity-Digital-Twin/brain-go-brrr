#!/usr/bin/env python3
"""Evaluate pretrained SeizureTransformer on TUSZ eval split.

Protocol:
- Use env vars for paths: BGB_DATA_ROOT (required)
- Window-level AUROC on mean probabilities per window
- No post-processing before AUROC

Expected AUROC: ~0.876 (±0.02) if weights and preprocessing match reference.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from brain_go_brrr.infra.data.tusz_detection_dataset import (
    TUSZDetectionDataset,
    WindowConfig,
)
from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import (
    SeizureTransformerWrapper,
)


def set_seeds(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_pretrained_model() -> float:
    set_seeds(42)

    data_root = Path(os.getenv("BGB_DATA_ROOT", ""))
    if not data_root:
        raise RuntimeError("BGB_DATA_ROOT env var is required")

    weights = data_root / "models/pretrained/seizure_transformer_wu2025.pth"
    tusz_root = data_root / "datasets/tusz/edf"

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not tusz_root.exists():
        raise FileNotFoundError(f"TUSZ root not found: {tusz_root}")

    # Create wrapper (loads model + pre/post processors)
    wrapper = SeizureTransformerWrapper(weights_path=weights)

    # Build eval dataset using SSOT preprocessor
    cfg = WindowConfig(fs=256, window_sec=60.0, stride_sec=60.0)
    ds = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="eval",
        cfg=cfg,
        preprocessor=wrapper.preprocessor,
        ensure_unipolar=True,
    )

    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)

    all_probs: list[float] = []
    all_labels: list[int] = []

    wrapper.model.eval()
    with torch.no_grad():
        device = wrapper.device
        for x, y in loader:
            # x: (B, 19, 15360) already preprocessed
            x = x.to(device)
            logits = wrapper.model(x)  # (B, 15360)
            probs = torch.sigmoid(logits).mean(dim=1)  # per-window mean
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y.numpy().astype(int).tolist())

    y_true = np.asarray(all_labels, dtype=np.int32)
    y_score = np.asarray(all_probs, dtype=np.float32)
    auroc = float(roc_auc_score(y_true, y_score))

    print(f"AUROC: {auroc:.3f} (expected ≈ 0.876)")
    return auroc


if __name__ == "__main__":
    evaluate_pretrained_model()
