# HOW TO INTEGRATE CHANNEL MAPPER IN train_tuev_mne.py

## Quick Integration Steps

### 1. Add import at top (around line 38):
```python
from brain_go_brrr.infra.ml_models.channel_mapper import TUEVChannelMapper
```

### 2. After creating probe (around line 475), add:
```python
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
```

### 3. Update optimizer (around line 478):
```python
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
```

### 4. In train_epoch function, add channel_mapper parameter:
```python
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
```

### 5. In training loop (around line 159), before feature extraction:
```python
# Apply channel mapper if provided
if channel_mapper is not None:
    x = channel_mapper(x)  # (B, 23, T) -> (B, 20, T)

# Then continue with existing feature extraction:
with torch.no_grad():
    features = model.extract_features(x, summary=False)
```

### 6. When calling train_epoch, pass channel_mapper:
```python
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
```

### 7. Similarly update evaluate() function to accept and use channel_mapper

## That's it! The mapper will transform 23→20 channels before EEGPT.