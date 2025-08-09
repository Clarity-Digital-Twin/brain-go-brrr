"""Dry run test to verify OneCycleLR scheduler is working correctly."""

import torch
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

def test_scheduler(config_path="configs/tuab_4s_paper_aligned.yaml"):
    """Test scheduler behavior without actually training."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Simulate training parameters
    batch_size = config['data']['batch_size']
    max_epochs = config['training']['max_epochs']
    accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    # Assume we have ~2900 batches per epoch (TUAB train set)
    batches_per_epoch = 2914  # Approximate from TUAB dataset
    
    # Calculate total OPTIMIZER steps (not batch steps)
    import math
    steps_per_epoch = math.ceil(batches_per_epoch / accum_steps)
    total_steps = steps_per_epoch * max_epochs
    
    print(f"Configuration:")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Gradient accumulation: {accum_steps}")
    print(f"  Optimizer steps per epoch: {steps_per_epoch}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Total optimizer steps: {total_steps}")
    print()
    
    # Create dummy optimizer
    dummy_param = torch.randn(10, 10, requires_grad=True)
    optimizer = torch.optim.AdamW([dummy_param], lr=0.001)
    
    # Create scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['training']['scheduler']['max_lr'],
        total_steps=total_steps,
        pct_start=config['training']['scheduler']['pct_start'],
        anneal_strategy=config['training']['scheduler']['anneal_strategy'],
        div_factor=config['training']['scheduler']['div_factor'],
        final_div_factor=config['training']['scheduler']['final_div_factor']
    )
    
    # Expected LR values
    max_lr = config['training']['scheduler']['max_lr']
    initial_lr = max_lr / config['training']['scheduler']['div_factor']
    final_lr = max_lr / config['training']['scheduler']['final_div_factor']
    warmup_steps = int(total_steps * config['training']['scheduler']['pct_start'])
    
    print(f"Expected Learning Rate Schedule:")
    print(f"  Initial LR: {initial_lr:.6f}")
    print(f"  Max LR: {max_lr:.6f}")
    print(f"  Final LR: {final_lr:.6f}")
    print(f"  Warmup steps: {warmup_steps} ({config['training']['scheduler']['pct_start']*100:.1f}% of training)")
    print()
    
    # Simulate training
    lrs = []
    for step in range(total_steps):
        # Dummy forward/backward
        loss = (dummy_param ** 2).sum()
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Scheduler step (CRITICAL: must be after optimizer.step())
        scheduler.step()
        
        # Record LR
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        
        # Log key points
        if step == 0:
            print(f"Step {step:5d}: LR = {current_lr:.6f} (start)")
        elif step == warmup_steps:
            print(f"Step {step:5d}: LR = {current_lr:.6f} (end of warmup)")
        elif step == total_steps // 2:
            print(f"Step {step:5d}: LR = {current_lr:.6f} (midpoint)")
        elif step == total_steps - 1:
            print(f"Step {step:5d}: LR = {current_lr:.6f} (final)")
        elif step % 1000 == 0:
            print(f"Step {step:5d}: LR = {current_lr:.6f}")
    
    # Validation checks
    print("\n✅ Validation Checks:")
    
    # Check warmup
    if lrs[0] < lrs[warmup_steps]:
        print("  ✓ Warmup working (LR increases)")
    else:
        print("  ✗ WARMUP BROKEN! LR not increasing")
    
    # Check peak
    peak_lr = max(lrs)
    if abs(peak_lr - max_lr) < 0.0001:
        print(f"  ✓ Peak LR correct ({peak_lr:.6f} ≈ {max_lr:.6f})")
    else:
        print(f"  ✗ Peak LR wrong ({peak_lr:.6f} ≠ {max_lr:.6f})")
    
    # Check annealing
    if lrs[-1] < lrs[warmup_steps]:
        print("  ✓ Annealing working (LR decreases)")
    else:
        print("  ✗ ANNEALING BROKEN! LR not decreasing")
    
    # Check final LR
    if abs(lrs[-1] - final_lr) < 0.0001:
        print(f"  ✓ Final LR correct ({lrs[-1]:.6f} ≈ {final_lr:.6f})")
    else:
        print(f"  ✗ Final LR wrong ({lrs[-1]:.6f} ≠ {final_lr:.6f})")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(lrs)
    plt.axvline(warmup_steps, color='r', linestyle='--', label='End of warmup')
    plt.axhline(max_lr, color='g', linestyle='--', alpha=0.5, label=f'Max LR ({max_lr:.4f})')
    plt.xlabel('Optimizer Step')
    plt.ylabel('Learning Rate')
    plt.title('OneCycleLR Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(lrs)
    plt.axvline(warmup_steps, color='r', linestyle='--', label='End of warmup')
    plt.xlabel('Optimizer Step')
    plt.ylabel('Learning Rate (log scale)')
    plt.title('OneCycleLR Schedule (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scheduler_test.png', dpi=100)
    print(f"\n📊 Schedule plot saved to scheduler_test.png")
    
    # Test resume functionality
    print("\n🔄 Testing Resume Functionality:")
    
    # Simulate resume from 50% progress
    resume_step = total_steps // 2
    
    # Create new optimizer and scheduler
    optimizer2 = torch.optim.AdamW([torch.randn(10, 10, requires_grad=True)], lr=0.001)
    scheduler2 = OneCycleLR(
        optimizer2,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=config['training']['scheduler']['pct_start'],
        anneal_strategy=config['training']['scheduler']['anneal_strategy'],
        div_factor=config['training']['scheduler']['div_factor'],
        final_div_factor=config['training']['scheduler']['final_div_factor'],
        last_epoch=resume_step - 1  # Resume from this step
    )
    
    # Step once and check LR
    optimizer2.step()
    scheduler2.step()
    resumed_lr = optimizer2.param_groups[0]['lr']
    expected_lr = lrs[resume_step]
    
    if abs(resumed_lr - expected_lr) < 0.0001:
        print(f"  ✓ Resume working! LR = {resumed_lr:.6f} (expected {expected_lr:.6f})")
    else:
        print(f"  ✗ RESUME BROKEN! LR = {resumed_lr:.6f} (expected {expected_lr:.6f})")
    
    return lrs


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/tuab_4s_paper_aligned.yaml"
    
    print("="*60)
    print("OneCycleLR Scheduler Test")
    print("="*60)
    print()
    
    lrs = test_scheduler(config_path)
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("If all checks passed, the scheduler is working correctly.")
    print("="*60)