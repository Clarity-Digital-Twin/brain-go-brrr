# OneCycleLR Scheduler Bug Analysis & Fix - COMPLETE ANALYSIS

**Date**: 2025-08-09  
**Updated**: After deep review and external feedback
**Issue**: Training plateaued with constant learning rate despite OneCycleLR configuration

## 🔴 MULTIPLE CRITICAL BUGS IDENTIFIED

### The Problem
Training run showed constant learning rate of 2.84e-03 throughout 77% of training (45,114/58,285 steps), causing:
- Loss oscillating between 0.38-0.65 without improvement
- No warmup phase (should start at ~0.0001)
- No annealing phase (should end at ~0.000003)
- Model stuck in local minimum due to high LR

### Root Causes (Multiple Issues Found!)

1. **Scheduler stepping per epoch instead of per batch** (train_paper_aligned_resume.py)
   ```python
   # Line 179 (WRONG):
   scheduler.step()  # Called OUTSIDE the batch loop!
   ```

2. **start_epoch reset bug** (train_paper_aligned_resume.py line 318):
   ```python
   steps_completed = start_epoch * steps_per_epoch  # Line 299
   # ... scheduler created with steps_completed ...
   start_epoch = 0  # Line 318 - RESETS IT! Breaking resume!
   ```

3. **Missing gradient accumulation awareness**:
   ```python
   # WRONG: Doesn't account for accumulation
   total_steps = len(train_loader) * max_epochs
   
   # CORRECT:
   accum_steps = config.get('gradient_accumulation_steps', 1)
   steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
   total_steps = steps_per_epoch * max_epochs
   ```

4. **No global_step tracking** for proper resume
5. **Using scheduler.get_last_lr()** instead of optimizer LR

## ✅ COMPREHENSIVE FIX

### Complete Fixed Implementation (train_paper_aligned_FINAL.py)
```python
def train_epoch(model, probe, train_loader, optimizer, scheduler, device, config):
    """Train for one epoch."""
    model.eval()  # Backbone stays frozen
    probe.train()
    
    losses = []
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, labels) in enumerate(pbar):
        data = data.to(device)
        labels = labels.to(device)
        
        # Forward through frozen backbone
        with torch.no_grad():
            features = model(data)
            
        # Forward through probe
        logits = probe(features)
        loss = F.cross_entropy(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            probe.parameters(), 
            config['training']['gradient_clip_val']
        )
        
        optimizer.step()
        
        # Gradient accumulation aware stepping
        should_step = ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(train_loader))
        
        if should_step:
            torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_val)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # ✅ CRITICAL: Step scheduler ONLY when optimizer steps
            scheduler.step()
            global_step += 1
        
        # ✅ Use OPTIMIZER LR (not scheduler)
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}',
            'step': global_step
        })
```

### For Gradient Accumulation (if used)
```python
accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

for batch_idx, (data, labels) in enumerate(pbar):
    # ... forward pass ...
    loss = loss / accumulation_steps
    loss.backward()
    
    if (batch_idx + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(probe.parameters(), clip_val)
        optimizer.step()
        scheduler.step()  # ✅ Step only when optimizer steps
        optimizer.zero_grad()
```

## 📊 Expected Learning Rate Schedule

With proper per-batch stepping:

| Phase | Steps | Learning Rate | Purpose |
|-------|-------|---------------|---------|
| Warmup (10%) | 0-5,828 | 0.0001 → 0.003 | Gradual increase |
| Peak (40%) | 5,829-29,142 | 0.003 | Maximum learning |
| Annealing (50%) | 29,143-58,285 | 0.003 → 0.000003 | Fine-tuning |

## 🔧 Fixed Scheduler Configuration

Complete setup with all fixes:
```python
# CRITICAL: Calculate total steps with accumulation awareness
import math

accum_steps = config['training'].get('gradient_accumulation_steps', 1)
steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
total_steps = steps_per_epoch * config['training']['max_epochs']

# Initialize tracking BEFORE resume logic
start_epoch = 0
global_step = 0
best_auroc = 0.0

# Resume logic (BEFORE scheduler creation)
if resume_checkpoint:
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint.get('epoch', 0) + 1
    global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)
    best_auroc = checkpoint.get('val_auroc', 0.0)
    # DO NOT reset start_epoch after this!

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config['training']['scheduler']['max_lr'],  # 0.003
    total_steps=total_steps,  # Adjusted for accumulation!
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
    div_factor=25,  # initial_lr = max_lr/25 = 0.00012
    final_div_factor=1000,  # final_lr = max_lr/1000 = 0.000003
    last_epoch=global_step - 1 if global_step > 0 else -1  # Resume support!
)
```

## 🔍 Critical Validation Checks

Essential validation to ensure everything works:
```python
# 1. Use optimizer LR (more reliable)
current_lr = optimizer.param_groups[0]['lr']

# 2. Track LR changes
if global_step > 10 and global_step % 100 == 0:
    if not hasattr(train_epoch, 'last_lr'):
        train_epoch.last_lr = current_lr
    elif abs(current_lr - train_epoch.last_lr) < 1e-10:
        logger.warning(f"⚠️ LR not changing! Stuck at {current_lr:.2e}")
    train_epoch.last_lr = current_lr

# 3. Verify total steps
if global_step > total_steps:
    logger.warning(f"global_step {global_step} > total_steps {total_steps}")

# 4. Save global_step in checkpoint
torch.save({
    'epoch': epoch,
    'global_step': global_step,  # CRITICAL!
    'probe_state_dict': probe.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    # ... other fields
}, checkpoint_path)
```

## 📈 Resume Support

When resuming training:
```python
# Load checkpoint
checkpoint = torch.load(resume_path)
start_epoch = checkpoint['epoch'] + 1
global_step = checkpoint.get('global_step', 0)

# Recreate scheduler with correct last_epoch
scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    total_steps=total_steps,
    last_epoch=global_step - 1,  # Critical for resume!
    # ... other params
)
```

## 🎯 Impact on Training

### Without Fix (Current)
- Constant high LR causes unstable training
- Model can't fine-tune in later stages
- AUROC plateaus around 0.85-0.86

### With Fix (Expected)
- Smooth warmup prevents early instability
- Peak LR for rapid learning
- Annealing allows fine-tuning
- AUROC should reach 0.869+ (paper target)

## 🚀 Action Items

1. **Current Run**: Let it finish (establish baseline)
2. **Next Run**: Apply scheduler fix to both scripts
3. **Verify**: Check LR changes every batch in logs
4. **Monitor**: Track AUROC improvement with proper annealing

## 🎯 Key Findings from Deep Review

### From EEGPT Paper
- Paper uses OneCycleLR with batch_size=64, 200 epochs
- Initial LR: 2.5e-4, Max: 5e-4, Min: 3.13e-5
- Linear probe only updates probe weights (backbone frozen)
- 4-second windows at 256Hz (1024 samples)
- Target AUROC for TUAB: 0.8718 ± 0.005

### Critical Implementation Details
1. **Scheduler must step per optimizer step** (not per batch if using accumulation)
2. **Total steps = optimizer steps** (not dataloader iterations)
3. **Track global_step** for proper resume
4. **Never reset start_epoch** after calculating resume position
5. **Use optimizer.param_groups[0]['lr']** for accurate LR monitoring

## 📝 Complete Verification Checklist

```python
# 1. Dry run test (before training)
def test_scheduler_dry_run(config, train_loader):
    """Test scheduler behavior without training."""
    import math
    
    # Setup
    accum = config.get('gradient_accumulation_steps', 1)
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * config['max_epochs']
    
    # Create dummy optimizer
    optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=0.003)
    scheduler = OneCycleLR(optimizer, max_lr=0.003, total_steps=total_steps)
    
    # Simulate training
    lrs = []
    for step in range(min(200, total_steps)):  # First 200 steps
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        lrs.append(lr)
        if step % 10 == 0:
            print(f"Step {step}: LR = {lr:.6f}")
    
    # Verify warmup is happening
    assert lrs[0] < lrs[50], "No warmup detected!"
    print("✅ Scheduler warmup verified")
    return lrs

# 2. Runtime verification
if epoch == 0 and batch_idx < 5:
    logger.info(f"Early training check - Step {global_step}: LR = {current_lr:.6f}")
    # Should see LR increasing from ~0.00012

# 3. Mid-training check
if global_step == total_steps // 2:
    expected_lr = config['scheduler']['max_lr']  # Should be at peak
    if abs(current_lr - expected_lr) > 0.0001:
        logger.warning(f"Mid-training LR mismatch: {current_lr:.6f} vs expected {expected_lr:.6f}")

# 4. Final sanity check
if global_step >= total_steps - 10:
    final_expected = config['scheduler']['max_lr'] / config['scheduler']['final_div_factor']
    logger.info(f"Near end - LR should be approaching {final_expected:.6f}, actual: {current_lr:.6f}")
```

---

**Remember**: OneCycleLR must step **per batch**, not per epoch!