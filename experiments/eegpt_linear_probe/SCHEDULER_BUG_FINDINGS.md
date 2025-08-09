# OneCycleLR Scheduler Bug Analysis & Fix

**Date**: 2025-08-09
**Issue**: Training plateaued with constant learning rate despite OneCycleLR configuration

## 🔴 CRITICAL BUG IDENTIFIED

### The Problem
Training run showed constant learning rate of 2.84e-03 throughout 77% of training (45,114/58,285 steps), causing:
- Loss oscillating between 0.38-0.65 without improvement
- No warmup phase (should start at ~0.0001)
- No annealing phase (should end at ~0.000003)
- Model stuck in local minimum due to high LR

### Root Cause
**Scheduler stepping per epoch instead of per batch!**

Current implementation in `train_paper_aligned.py` and `train_paper_aligned_resume.py`:
```python
# Line 179-180 (WRONG):
scheduler.step()  # Called OUTSIDE the batch loop!
```

This causes OneCycleLR to advance only ~20 times instead of ~58,000 times!

## ✅ THE FIX

### Correct Implementation
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
        
        # ✅ CRITICAL FIX: Step scheduler after EACH optimizer step
        scheduler.step()
        
        # Update progress bar with CURRENT learning rate
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'  # Should change every batch!
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

## 🔧 Scheduler Configuration

Correct OneCycleLR setup:
```python
# Calculate total steps correctly
steps_per_epoch = len(train_loader)
total_steps = steps_per_epoch * config['training']['max_epochs']

# For gradient accumulation:
# total_steps = (steps_per_epoch // accumulation_steps) * max_epochs

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=config['training']['scheduler']['max_lr'],  # 0.003
    total_steps=total_steps,  # ~58,285 for 20 epochs
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos',
    div_factor=25,  # initial_lr = max_lr/25 = 0.00012
    final_div_factor=1000,  # final_lr = max_lr/1000 = 0.000003
)
```

## 🔍 Debugging & Verification

Add these checks to ensure scheduler works:
```python
# Log every N steps
if batch_idx % 100 == 0:
    current_lr = scheduler.get_last_lr()[0]
    logger.info(f"Step {scheduler._step_count}: LR = {current_lr:.6f}")
    
    # Verify LR is changing
    if batch_idx > 0 and abs(current_lr - prev_lr) < 1e-8:
        logger.warning("LR not changing! Check scheduler.step() placement")
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

## 📝 Testing the Fix

Quick test to verify scheduler:
```python
# Test scheduler behavior
optimizer = torch.optim.AdamW([torch.randn(10, 10)], lr=0.003)
scheduler = OneCycleLR(optimizer, max_lr=0.003, total_steps=1000)

lrs = []
for i in range(1000):
    optimizer.step()
    scheduler.step()
    lrs.append(scheduler.get_last_lr()[0])

# Plot should show: rise → plateau → decay
import matplotlib.pyplot as plt
plt.plot(lrs)
plt.ylabel('Learning Rate')
plt.xlabel('Step')
plt.title('OneCycleLR Schedule')
plt.show()
```

---

**Remember**: OneCycleLR must step **per batch**, not per epoch!