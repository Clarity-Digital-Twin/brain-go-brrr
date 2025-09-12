#!/usr/bin/env python3
"""Evaluate pretrained SeizureTransformer on TUSZ test set.

This script implements the exact evaluation protocol from Wu et al. 2025:
- Window-level AUROC computation (mean probability per window)
- No post-processing before AUROC calculation
- Uses environment variables for paths
- Expected AUROC: 0.876 ± 0.02 on TUSZ eval set

Usage:
    export BGB_DATA_ROOT="/path/to/data"
    export BGB_OUTPUT_ROOT="./experiments/seizure_transformer/runs"
    python scripts/evaluate_seizure_transformer.py
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from src - clean architecture
from brain_go_brrr.infra.data.tusz_detection_dataset import (
    TUSZDetectionDataset,
    WindowConfig,
)
from brain_go_brrr.infra.ml_models.seizure_transformer_wrapper import (
    SeizureTransformerWrapper,
)


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_pretrained_model(
    weights_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict:
    """Evaluate pretrained SeizureTransformer on TUSZ eval set.
    
    Args:
        weights_path: Path to pretrained weights (uses env if None)
        output_dir: Directory for outputs (uses env if None)
        batch_size: Batch size for evaluation
        num_workers: Number of dataloader workers
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("SEIZURE TRANSFORMER EVALUATION")
    print("Using pretrained weights on TUSZ eval set")
    print("=" * 60)
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Get paths from environment
    data_root = Path(
        os.getenv(
            "BGB_DATA_ROOT",
            "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brrr/data"
        )
    )
    
    if weights_path is None:
        weights_path = data_root / "models/pretrained/seizure_transformer_wu2025.pth"
    
    if output_dir is None:
        output_root = Path(
            os.getenv(
                "BGB_OUTPUT_ROOT",
                "./experiments/seizure_transformer/runs"
            )
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"eval_{timestamp}"
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    
    # Log configuration
    config = {
        "weights_path": str(weights_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": 42,
    }
    
    with open(output_dir / "configs" / "eval_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 1. Load pretrained model
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at {weights_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wrapper with pretrained weights
    wrapper = SeizureTransformerWrapper(
        weights_path=weights_path,
        device=device,
    )
    print(f"✅ Loaded pretrained weights from {weights_path}")
    
    # 2. Create test dataset (TUSZ eval split ONLY!)
    tusz_root = data_root / "datasets/tusz/edf"
    if not tusz_root.exists():
        raise FileNotFoundError(f"TUSZ dataset not found at {tusz_root}")
    
    print("\nLoading TUSZ eval dataset...")
    test_dataset = TUSZDetectionDataset(
        root_dir=tusz_root,
        split="eval",  # CRITICAL: Use eval split only!
        cfg=WindowConfig(
            fs=256,
            window_sec=60.0,
            stride_sec=60.0,  # No overlap for inference
            positive_fraction=0.5,  # Use 0.5 threshold for binary labels
        ),
        target_channels=None,  # Use default 19 channels
        max_windows=None,  # Process all windows
    )
    print(f"✅ Loaded TUSZ eval set: {len(test_dataset)} windows")
    
    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 3. Run inference
    print("\nRunning inference...")
    all_probs = []
    all_labels = []
    
    wrapper.model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch_data = batch_data.to(device)  # (B, 19, 15360)
            
            # Run model - outputs per-timestep predictions
            logits = wrapper.model(batch_data)  # (B, 15360)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            
            # For window-level AUROC: mean probability per window
            window_probs = probs.mean(dim=1)  # (B,)
            
            # Store predictions and labels
            all_probs.extend(window_probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Convert to arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 4. Calculate metrics (NO post-processing for AUROC!)
    print("\nCalculating metrics...")
    
    # Handle edge case where only one class is present
    unique_labels = np.unique(all_labels)
    if len(unique_labels) == 1:
        print(f"⚠️ Warning: Only one class in eval set (all {unique_labels[0]}s)")
        auroc = 0.5
    else:
        # Calculate AUROC on raw probabilities
        auroc = roc_auc_score(all_labels, all_probs)
        
        # Calculate ROC curve for operating points
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    # Calculate class distribution
    n_positive = all_labels.sum()
    n_negative = len(all_labels) - n_positive
    positive_ratio = n_positive / len(all_labels)
    
    # 5. Apply post-processing for clinical metrics only
    print("\nApplying post-processing for clinical metrics...")
    
    # Create full probability array for post-processing
    # (This would normally be done on the full time-series, not window means)
    post_processed = wrapper.postprocessor.postprocess(all_probs)
    
    # Calculate post-processing metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(all_labels, post_processed > 0.5)
    precision = precision_score(all_labels, post_processed > 0.5, zero_division=0)
    recall = recall_score(all_labels, post_processed > 0.5, zero_division=0)
    f1 = f1_score(all_labels, post_processed > 0.5, zero_division=0)
    
    # 6. Compile results
    results = {
        "auroc": float(auroc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "n_windows": len(all_labels),
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "positive_ratio": float(positive_ratio),
        "expected_auroc": 0.876,
        "auroc_diff": float(abs(auroc - 0.876)),
    }
    
    # Save metrics
    with open(output_dir / "metrics" / "eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 7. Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"AUROC: {auroc:.4f} (Expected: 0.876)")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nDataset Statistics:")
    print(f"  Total windows: {len(all_labels):,}")
    print(f"  Positive windows: {n_positive:,} ({100*positive_ratio:.1f}%)")
    print(f"  Negative windows: {n_negative:,} ({100*(1-positive_ratio):.1f}%)")
    
    # Success check
    if abs(auroc - 0.876) < 0.02:
        print("\n✅ SUCCESS: AUROC matches paper within tolerance!")
    else:
        print(f"\n⚠️ WARNING: AUROC differs from paper by {abs(auroc - 0.876):.3f}")
    
    # 8. Generate summary report
    report = f"""
SEIZURE TRANSFORMER EVALUATION REPORT
=====================================
Generated: {datetime.now().isoformat()}

Model Performance:
-----------------
AUROC: {auroc:.4f} (Target: 0.876)
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}

Dataset:
--------
Total Windows: {len(all_labels):,}
Positive: {n_positive:,} ({100*positive_ratio:.1f}%)
Negative: {n_negative:,} ({100*(1-positive_ratio):.1f}%)

Configuration:
-------------
Weights: {weights_path}
Device: {device}
Batch Size: {batch_size}
Workers: {num_workers}

Status: {'✅ PASS' if abs(auroc - 0.876) < 0.02 else '⚠️ NEEDS INVESTIGATION'}
"""
    
    with open(output_dir / "reports" / "summary.txt", "w") as f:
        f.write(report)
    
    print(f"\n📊 Results saved to: {output_dir}")
    
    return results


def main():
    """Main entry point for evaluation script."""
    try:
        results = evaluate_pretrained_model()
        
        # Exit code based on success
        if abs(results["auroc"] - 0.876) < 0.02:
            exit(0)  # Success
        else:
            exit(1)  # Performance not matching paper
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(2)  # Error during evaluation


if __name__ == "__main__":
    main()