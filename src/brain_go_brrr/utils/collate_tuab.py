"""TUAB-specific collate function - enforces 19 channels with temporary workaround."""

import torch


def collate_tuab_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor | int | float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for TUAB dataset - expects 19 channels.

    TUAB specifications:
    - Expects exactly 19 channels (no Fz)
    - BCEWithLogitsLoss expects float labels (0.0, 1.0)

    TEMPORARY WORKAROUND:
    - Handles 304 contaminated windows with 20 channels by dropping Fz (index 4)
    - This workaround will be removed after cache is fixed

    Args:
        batch: List of (eeg_data, label) tuples from DataLoader

    Returns:
        Tuple of (batch_data, batch_labels)
        - batch_data: (B, 19, 1024) tensor
        - batch_labels: (B,) tensor with float32 dtype
    """
    # Stack data - handle 19 vs 20 channel mismatch by truncating to 19
    # TEMPORARY: This handles the 304 contaminated windows
    processed_samples = []
    for idx, (x, _) in enumerate(batch):
        if x.shape[0] == 20:
            # Drop Fz channel (typically channel 4) to get 19 channels
            # Standard 10-20 order: Fp1, Fp2, F7, F3, Fz, F4, F8, ...
            # We want to exclude Fz (index 4)
            # NOTE: This is a WORKAROUND for 304 bad windows from cache v2
            x = torch.cat([x[:4], x[5:]], dim=0)  # Skip channel 4 (Fz)
        elif x.shape[0] == 19:
            # Correct shape - no modification needed
            pass
        else:
            raise RuntimeError(
                f"TUAB batch item {idx}: Unexpected channel count {x.shape[0]}. "
                f"Expected 19 (or 20 with workaround). Shape: {x.shape}"
            )
        processed_samples.append(x)

    data = torch.stack(processed_samples)

    # Handle labels - TUAB uses float labels for BCEWithLogitsLoss
    first_label = batch[0][1]

    if isinstance(first_label, torch.Tensor):
        # Labels are already tensors - stack and preserve dtype
        labels = torch.stack([sample[1] for sample in batch])
        # Ensure shape is (B,) not (B, 1) for compatibility
        labels = labels.view(-1).float()  # Ensure float for BCE loss
    else:
        # Labels are Python scalars - convert to float32 for TUAB
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)

    return data, labels
