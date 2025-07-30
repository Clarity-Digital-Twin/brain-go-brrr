#!/usr/bin/env python3
"""Trace through EEGPT forward pass to find where features become non-discriminative."""

import numpy as np
import torch

from brain_go_brrr.models.eegpt_architecture import create_eegpt_model


def trace_forward():
    """Trace step by step through forward pass."""
    print("=== TRACING EEGPT FORWARD PASS ===\n")

    # Load model
    model = create_eegpt_model("data/models/eegpt/pretrained/eegpt_mcae_58chs_4s_large4E.ckpt")
    model.eval()

    # Create very different inputs - just 2 channels for simplicity
    t = np.arange(1024) / 256

    # Input 1: Sine wave on all channels
    sine = np.sin(2 * np.pi * 10 * t) * 50e-6
    input1 = torch.FloatTensor(np.tile(sine, (2, 1))).unsqueeze(0)

    # Input 2: Random noise
    input2 = torch.randn(1, 2, 1024) * 50e-6

    print(f"Input shape: {input1.shape}")

    # Manual forward pass
    with torch.no_grad():
        # Step 1: Patch embedding
        x1 = model.patch_embed(input1)
        x2 = model.patch_embed(input2)
        B, N, C, D = x1.shape
        print(f"\n1. After patch embed: shape {x1.shape}")
        sim = torch.nn.functional.cosine_similarity(
            x1.flatten().unsqueeze(0), x2.flatten().unsqueeze(0)
        ).item()
        print(f"   Similarity: {sim:.6f}")

        # Step 2: Channel IDs and embedding
        chan_ids = torch.arange(0, C, dtype=torch.long)
        chan_embed = model.chan_embed(chan_ids).unsqueeze(0).unsqueeze(0)
        print(f"\n2. Channel embed shape: {chan_embed.shape}")

        # Step 3: Add channel embedding
        x1 = x1 + chan_embed
        x2 = x2 + chan_embed
        print("\n3. After adding channel embed:")
        sim = torch.nn.functional.cosine_similarity(
            x1.flatten().unsqueeze(0), x2.flatten().unsqueeze(0)
        ).item()
        print(f"   Similarity: {sim:.6f}")

        # Step 4: Reshape for transformer
        x1 = x1.reshape(B, N * C, D)
        x2 = x2.reshape(B, N * C, D)
        print(f"\n4. After reshape: shape {x1.shape}")

        # Step 5: Add summary tokens
        summary_tokens = model.summary_token.repeat(B, 1, 1)
        x1 = torch.cat([x1, summary_tokens], dim=1)
        x2 = torch.cat([x2, summary_tokens], dim=1)
        print(f"\n5. After adding summary tokens: shape {x1.shape}")
        sim = torch.nn.functional.cosine_similarity(
            x1.flatten().unsqueeze(0), x2.flatten().unsqueeze(0)
        ).item()
        print(f"   Similarity: {sim:.6f}")

        # Step 6: Through transformer blocks
        for i, block in enumerate(model.blocks):
            x1 = block(x1)
            x2 = block(x2)
            if i == 0 or i == len(model.blocks) - 1:
                sim = torch.nn.functional.cosine_similarity(
                    x1.flatten().unsqueeze(0), x2.flatten().unsqueeze(0)
                ).item()
                print(f"\n6.{i + 1} After block {i}: similarity {sim:.6f}")

        # Step 7: Extract summary tokens
        x1 = x1[:, -model.embed_num :, :]
        x2 = x2[:, -model.embed_num :, :]
        print(f"\n7. After extracting summary tokens: shape {x1.shape}")
        sim = torch.nn.functional.cosine_similarity(
            x1.flatten().unsqueeze(0), x2.flatten().unsqueeze(0)
        ).item()
        print(f"   Similarity: {sim:.6f}")

        # Step 8: Final norm
        x1 = model.norm(x1)
        x2 = model.norm(x2)
        print("\n8. After final norm:")
        sim = torch.nn.functional.cosine_similarity(
            x1.flatten().unsqueeze(0), x2.flatten().unsqueeze(0)
        ).item()
        print(f"   Similarity: {sim:.6f}")

        # Check summary token values
        print("\n=== Summary Token Analysis ===")
        print(
            f"Summary token 0 - Input 1 mean: {x1[0, 0, :].mean():.6f}, std: {x1[0, 0, :].std():.6f}"
        )
        print(
            f"Summary token 0 - Input 2 mean: {x2[0, 0, :].mean():.6f}, std: {x2[0, 0, :].std():.6f}"
        )

        # Check attention weights from first block
        print("\n=== Attention Weight Check ===")
        print(
            f"First block attention QKV weight norm: {model.blocks[0].attn.qkv.weight.norm():.3f}"
        )
        print(
            f"First block attention proj weight norm: {model.blocks[0].attn.proj.weight.norm():.3f}"
        )


if __name__ == "__main__":
    trace_forward()
