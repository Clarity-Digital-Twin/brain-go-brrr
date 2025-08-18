#!/usr/bin/env python
"""Final TUEV Training Verification Script - Should We Continue or Abort?"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

def check_training_status():
    """Check if TUEV training with full features should continue."""
    
    print("=" * 80)
    print("TUEV TRAINING VERIFICATION REPORT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # 1. Check current metrics from logs
    log_file = Path("logs/tuev_FIXED_seed42.log")
    if not log_file.exists():
        print("❌ ERROR: Log file not found!")
        return False
    
    # Parse latest metrics
    with open(log_file) as f:
        lines = f.readlines()
    
    # Find epoch 1 results - look for first validation line
    epoch1_metrics = {}
    found_epoch1 = False
    for line in lines:
        # Look for Val line after Epoch 001
        if "Epoch 001:" in line:
            found_epoch1 = True
        if found_epoch1 and "Val" in line and "BAcc:" in line:
            # Extract metrics from line
            if "BAcc:" in line:
                bacc_str = line.split("BAcc:")[1].split(",")[0].strip()
                epoch1_metrics['bacc'] = float(bacc_str)
            if "F1:" in line:
                f1_str = line.split("F1:")[1].split(",")[0].strip()
                epoch1_metrics['f1'] = float(f1_str)
            if "Loss:" in line:
                loss_str = line.split("Loss:")[1].split(",")[0].strip()
                epoch1_metrics['loss'] = float(loss_str)
            if "Kappa:" in line:
                kappa_str = line.split("Kappa:")[1].strip()
                epoch1_metrics['kappa'] = float(kappa_str)
            break
    
    print("EPOCH 1 VALIDATION RESULTS:")
    bacc = epoch1_metrics.get('bacc', None)
    f1 = epoch1_metrics.get('f1', None)
    kappa = epoch1_metrics.get('kappa', None)
    loss = epoch1_metrics.get('loss', None)
    
    bacc_str = f"{bacc:.4f}" if bacc is not None else "N/A"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
    kappa_str = f"{kappa:.4f}" if kappa is not None else "N/A"
    loss_str = f"{loss:.4f}" if loss is not None else "N/A"
    
    print(f"  Balanced Accuracy: {bacc_str}")
    print(f"  Weighted F1:       {f1_str}")
    print(f"  Cohen's Kappa:     {kappa_str}")
    print(f"  Loss:              {loss_str}")
    print()
    
    print("PAPER TARGETS:")
    print("  Balanced Accuracy: 0.6232 ± 0.0114")
    print("  Weighted F1:       0.8187 ± 0.0063")
    print("  Cohen's Kappa:     0.6351 ± 0.0134")
    print()
    
    # 2. Apply dossier criteria
    print("DOSSIER CRITERIA CHECK:")
    
    criteria_passed = True
    
    # Check loss threshold
    loss_val = epoch1_metrics.get('loss', 100)
    if loss_val > 5.0:
        print(f"  ❌ Loss > 5.0 after epoch 1 ({loss_val:.2f})")
        criteria_passed = False
    else:
        print(f"  ✅ Loss < 5.0 ({loss_val:.2f})")
    
    # Check BAcc threshold
    bacc_val = epoch1_metrics.get('bacc', 0)
    if bacc_val < 0.30:
        print(f"  ❌ BAcc < 0.30 ({bacc_val:.4f})")
        criteria_passed = False
    else:
        print(f"  ✅ BAcc > 0.30 ({bacc_val:.4f})")
    
    # Check if worse than random
    random_bacc = 1/6  # 0.167 for 6 classes
    if epoch1_metrics.get('bacc', 0) < random_bacc:
        print(f"  ❌ CRITICAL: BAcc WORSE than random ({random_bacc:.3f})")
        criteria_passed = False
    
    # Check negative kappa
    if epoch1_metrics.get('kappa', 0) < 0:
        print(f"  ❌ CRITICAL: Negative Kappa = worse than chance!")
        criteria_passed = False
    
    print()
    print("=" * 80)
    print("ARCHITECTURAL ANALYSIS:")
    print("=" * 80)
    
    print("CURRENT APPROACH: Using ALL patch features")
    print("  - Feature dimensions: 163,840 (16 patches × 20 channels × 512)")
    print("  - Training samples: 83,932")
    print("  - Feature/sample ratio: 1.95 (TERRIBLE - should be << 1)")
    print()
    
    print("WHY IT'S FAILING:")
    print("  1. Too many features for linear classifier (163k params)")
    print("  2. Severe overfitting risk")
    print("  3. Not how EEGPT was designed to be used")
    print("  4. Gradient instability from massive feature space")
    print()
    
    print("RECOMMENDED FIX: Channel-pooled features")
    print("  - Feature dimensions: 10,240 (20 channels × 512)")
    print("  - Feature/sample ratio: 0.12 (much better)")
    print("  - Preserves channel-specific patterns for TUEV events")
    print()
    
    # 3. Final recommendation
    print("=" * 80)
    print("FINAL RECOMMENDATION:")
    print("=" * 80)
    
    if criteria_passed:
        print("✅ Training can continue (but unlikely to succeed)")
        recommendation = "CONTINUE"
    else:
        print("❌ ABORT IMMEDIATELY - Training has failed")
        print()
        print("ACTION ITEMS:")
        print("1. Kill the tmux session: tmux kill-session -t tuev_fixed_42")
        print("2. Implement proper feature extraction in src/brain_go_brrr/models/")
        print("3. Use channel-pooled features (10,240 dims)")
        print("4. Restart training with new architecture")
        recommendation = "ABORT"
    
    print()
    print("=" * 80)
    
    # Save verification results
    results = {
        'timestamp': datetime.now().isoformat(),
        'epoch1_metrics': epoch1_metrics,
        'criteria_passed': criteria_passed,
        'recommendation': recommendation,
        'feature_dims': 163840,
        'training_samples': 83932
    }
    
    with open('tuev_verification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to tuev_verification_results.json")
    
    return criteria_passed


def check_alternative_approaches():
    """Show alternative feature extraction approaches."""
    
    print("\n" + "=" * 80)
    print("ALTERNATIVE APPROACHES TO TRY:")
    print("=" * 80)
    
    approaches = [
        {
            'name': 'Summary Tokens (Original)',
            'dims': 2048,
            'description': '4 summary tokens × 512 dims',
            'pros': 'How EEGPT was designed',
            'cons': 'Already failed for TUEV (BAcc 0.16)'
        },
        {
            'name': 'Channel-Pooled Features',
            'dims': 10240,
            'description': 'Pool patches by channel: 20 × 512',
            'pros': 'Good for channel-specific events (SPSW, GPED)',
            'cons': 'Loses temporal resolution'
        },
        {
            'name': 'Temporal-Pooled Features',
            'dims': 8192,
            'description': 'Pool patches by time: 16 × 512',
            'pros': 'Good for temporal dynamics',
            'cons': 'Loses channel specificity'
        },
        {
            'name': 'Attention-Weighted Features',
            'dims': 'Variable',
            'description': 'Learn attention over patches',
            'pros': 'Adaptive feature selection',
            'cons': 'More complex, needs tuning'
        },
        {
            'name': 'PCA Reduced Features',
            'dims': 5000,
            'description': 'PCA on full features → 5k dims',
            'pros': 'Data-driven compression',
            'cons': 'Loses interpretability'
        }
    ]
    
    for approach in approaches:
        print(f"\n{approach['name']}:")
        print(f"  Dimensions: {approach['dims']}")
        print(f"  Method: {approach['description']}")
        print(f"  ✅ {approach['pros']}")
        print(f"  ⚠️  {approach['cons']}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED NEXT STEP: Try Channel-Pooled Features (10,240 dims)")
    print("=" * 80)


if __name__ == "__main__":
    # Run verification
    should_continue = check_training_status()
    
    # Show alternatives
    check_alternative_approaches()
    
    # Exit with appropriate code
    if not should_continue:
        print("\n🛑 TRAINING SHOULD BE ABORTED")
        exit(1)
    else:
        print("\n✅ Training can continue (but unlikely to succeed)")
        exit(0)
