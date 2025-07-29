"""Test what the encoder actually outputs to find summary tokens."""

import numpy as np
import torch

from src.brain_go_brrr.models.eegpt_model import EEGPTModel


def test_encoder_raw_output():
    """Check the raw encoder output to understand its structure."""
    model = EEGPTModel()
    ch_names = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "T3",
        "C3",
        "Cz",
        "C4",
        "T4",
        "T5",
        "P3",
        "Pz",
        "P4",
        "T6",
        "O1",
        "O2",
    ]

    # Simple test data
    data = np.random.randn(19, 1024) * 50e-6

    # Prepare data like in extract_features
    patch_size = 64
    n_patches = 1024 // patch_size  # 16
    n_channels = 19

    # Reshape to patches
    data_patched = data.reshape(n_channels, n_patches, patch_size)
    data_rearranged = np.transpose(data_patched, (1, 0, 2))  # (n_patches, n_channels, patch_size)
    data_flattened = data_rearranged.reshape(-1, patch_size)  # (n_patches * n_channels, patch_size)

    print(f"Data shape: {data.shape}")
    print(f"Patched shape: {data_patched.shape}")
    print(f"Rearranged shape: {data_rearranged.shape}")
    print(f"Flattened shape: {data_flattened.shape}")

    # Convert to tensor
    data_tensor = torch.FloatTensor(data_flattened).unsqueeze(0).to(model.device)
    print(f"Input tensor shape: {data_tensor.shape}")

    # Get channel IDs
    chan_ids = model._get_cached_channel_ids(ch_names)
    print(f"Channel IDs shape: {chan_ids.shape}")

    # Run through encoder
    with torch.no_grad():
        # Get raw encoder output
        encoder_output = model.encoder(data_tensor, chan_ids)
        print(f"\nEncoder output shape: {encoder_output.shape}")
        print(f"Encoder output dimensions: {encoder_output.dim()}")

        # Let's examine the structure
        if encoder_output.dim() == 3:
            batch, seq_len, embed_dim = encoder_output.shape
            print(f"  Batch size: {batch}")
            print(f"  Sequence length: {seq_len}")
            print(f"  Embedding dimension: {embed_dim}")

            # Expected: seq_len = n_patches * n_channels = 16 * 19 = 304
            expected_seq_len = n_patches * n_channels
            print(f"  Expected sequence length: {expected_seq_len}")

            # Check if there are extra tokens (summary tokens)
            extra_tokens = seq_len - expected_seq_len
            print(f"  Extra tokens (potential summary): {extra_tokens}")

            # Look at the first few tokens
            print("\nFirst 10 token statistics:")
            for i in range(min(10, seq_len)):
                token = encoder_output[0, i, :].cpu().numpy()
                print(
                    f"  Token {i}: mean={token.mean():.6f}, std={token.std():.6f}, min={token.min():.6f}, max={token.max():.6f}"
                )

        # Check what extract_features is doing
        print("\n--- Current extract_features logic ---")
        features = encoder_output.squeeze(0)  # Remove batch dim
        print(f"After squeeze: {features.shape}")

        # Current wrong approach
        features_reshaped = features.reshape(n_patches, n_channels, -1)
        print(f"After reshape: {features_reshaped.shape}")

        summary_features = features_reshaped.mean(dim=(0, 1))
        print(f"After averaging: {summary_features.shape}")

        summary_expanded = summary_features.unsqueeze(0).expand(4, -1)
        print(f"After expand: {summary_expanded.shape}")

        # This is why all 4 "summary tokens" are identical!
        print(
            f"\nAre expanded tokens identical? {torch.allclose(summary_expanded[0], summary_expanded[1])}"
        )


def test_find_summary_tokens():
    """Try to find where the summary tokens are in the encoder output."""
    model = EEGPTModel()

    # Check if encoder has summary tokens as parameters
    print("Checking encoder parameters for summary tokens:")
    for name, param in model.encoder.named_parameters():
        if "summary" in name.lower() or "cls" in name.lower() or "token" in name.lower():
            print(f"  Found: {name} - shape: {param.shape}")

    # Check encoder attributes
    print("\nChecking encoder attributes:")
    for attr in dir(model.encoder):
        if (
            "summary" in attr.lower() or "cls" in attr.lower() or "token" in attr.lower()
        ) and not attr.startswith("_"):
            print(f"  Found attribute: {attr}")
            try:
                value = getattr(model.encoder, attr)
                if hasattr(value, "shape"):
                    print(f"    Shape: {value.shape}")
                elif isinstance(value, int | float | str):
                    print(f"    Value: {value}")
            except Exception:
                pass


if __name__ == "__main__":
    test_encoder_raw_output()
    print("\n" + "=" * 50 + "\n")
    test_find_summary_tokens()
