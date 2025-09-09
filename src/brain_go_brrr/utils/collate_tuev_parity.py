"""TUEV paper parity collate function - accepts 23 channels for mapper."""

import torch


def collate_tuev_parity_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor | int | float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate function for TUEV paper parity mode - accepts 23 channels.

    Paper parity mode:
    - Accepts exactly 23 channels (includes A1, A2, T1, T2)
    - These will be mapped to 20 channels by TUEVChannelMapper in training
    - CrossEntropyLoss expects long labels (0, 1, 2, 3, 4, 5)

    Args:
        batch: List of (eeg_data, label) tuples from DataLoader

    Returns:
        Tuple of (batch_data, batch_labels)
        - batch_data: (B, 23, 1024) tensor
        - batch_labels: (B,) tensor with long dtype

    Raises:
        RuntimeError: If any sample doesn't have exactly 23 channels
    """
    # STRICT validation - Paper parity mode requires exactly 23 channels
    processed_samples = []
    for idx, (x, _) in enumerate(batch):
        if x.shape[0] != 23:
            raise RuntimeError(
                f"TUEV parity batch item {idx}: CHANNEL COUNT ERROR! "
                f"Got {x.shape[0]} channels, but paper parity requires EXACTLY 23. "
                f"Shape: {x.shape}. "
                f"Ensure use_paper_parity=True in dataset and cache was built correctly."
            )
        if x.shape[1] != 1024:
            raise RuntimeError(
                f"TUEV parity batch item {idx}: Sample count error! "
                f"Got {x.shape[1]} samples, expected 1024 (4s @ 256Hz). "
                f"Shape: {x.shape}"
            )
        processed_samples.append(x)

    data = torch.stack(processed_samples)

    # Handle labels - TUEV uses long labels for CrossEntropyLoss
    first_label = batch[0][1]

    if isinstance(first_label, torch.Tensor):
        # Labels are already tensors - stack and preserve dtype
        labels = torch.stack(
            [
                sample[1] if isinstance(sample[1], torch.Tensor) else torch.tensor(sample[1])
                for sample in batch
            ]
        )
        # Ensure shape is (B,) not (B, 1) for compatibility
        labels = labels.view(-1).long()  # Ensure long for CE loss
    else:
        # Labels are Python scalars - convert to long for TUEV
        # TUEV has 6 classes: SPSW(0), GPED(1), PLED(2), EYEM(3), ARTF(4), BCKG(5)
        labels = torch.tensor([sample[1] for sample in batch], dtype=torch.long)

        # Validate label range
        if labels.max() > 5 or labels.min() < 0:
            raise ValueError(
                f"TUEV labels out of range! Got min={labels.min()}, max={labels.max()}. "
                f"Expected 0-5 for 6 classes: SPSW, GPED, PLED, EYEM, ARTF, BCKG"
            )

    return data, labels
