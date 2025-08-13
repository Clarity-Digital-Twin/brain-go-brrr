#!/usr/bin/env python
"""Fix for robust abnormality detector loading."""

import os
import torch
import inspect
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

def load_checkpoint_robust(checkpoint_path: Path, device: str = "cpu") -> Dict[str, Any]:
    """Robustly load a checkpoint, handling various formats.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        
    Returns:
        Loaded checkpoint dictionary
    """
    if not checkpoint_path.exists():
        # Check env override
        env_model = os.getenv("BGB_ABN_MODEL")
        if env_model:
            checkpoint_path = Path(env_model)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Model not found at env path: {env_model}")
        else:
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    # Load with proper settings for PyTorch 2.6+
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        # MUST be False for checkpoints with numpy arrays
        kwargs["weights_only"] = False
    
    checkpoint = torch.load(checkpoint_path, **kwargs)
    return checkpoint

def extract_probe_weights(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Extract probe weights from various checkpoint formats.
    
    Handles:
    - Raw state_dict
    - Lightning format with 'state_dict' key
    - Our training format with 'probe_state_dict' key
    - DDP format with 'module.' prefix
    """
    # Our training format
    if "probe_state_dict" in checkpoint:
        state_dict = checkpoint["probe_state_dict"]
    # Lightning format
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    # Raw state dict
    elif isinstance(checkpoint, dict) and any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())[:10]}")
    
    # Strip common prefixes
    cleaned = {}
    for key, value in state_dict.items():
        # Remove DDP/Lightning prefixes
        for prefix in ["module.", "model.", "net.", "probe."]:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        cleaned[key] = value
    
    return cleaned

def build_probe_from_checkpoint(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    """Build a probe model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load to
        
    Returns:
        Loaded probe model
    """
    checkpoint = load_checkpoint_robust(checkpoint_path, device)
    
    # Extract config if available
    config = checkpoint.get("config", {})
    probe_config = config.get("model", {}).get("probe", {})
    
    # Get dimensions from config or infer from weights
    state_dict = extract_probe_weights(checkpoint)
    
    # Infer architecture from weights
    if "0.weight" in state_dict:
        # Linear probe format: 0.weight, 0.bias, 3.weight, 3.bias
        input_dim = state_dict["0.weight"].shape[1]  # [hidden, input]
        hidden_dim = state_dict["0.weight"].shape[0]
        output_dim = state_dict["3.weight"].shape[0] if "3.weight" in state_dict else 2
        
        # Build model
        probe = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(probe_config.get("dropout", 0.1)),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    else:
        raise ValueError(f"Cannot infer architecture from keys: {list(state_dict.keys())}")
    
    # Load weights
    probe.load_state_dict(state_dict)
    probe.to(device)
    probe.eval()
    
    # Log metrics if available
    if "val_auroc" in checkpoint:
        logger.info(f"Loaded probe with val_auroc: {checkpoint['val_auroc']:.4f}")
    if "epoch" in checkpoint:
        logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
    
    return probe

# Test the functions
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    # Test with our trained model
    model_path = Path("experiments/eegpt_linear_probe/output/tuab_4s_paper_target_BULLETPROOF_20250809_073159/best_model.pt")
    
    print("Testing robust checkpoint loading...")
    
    try:
        # Test loading
        checkpoint = load_checkpoint_robust(model_path)
        print(f"✅ Checkpoint loaded. Keys: {list(checkpoint.keys())}")
        
        # Test weight extraction
        weights = extract_probe_weights(checkpoint)
        print(f"✅ Weights extracted. Layers: {list(weights.keys())}")
        
        # Test probe building
        probe = build_probe_from_checkpoint(model_path)
        print(f"✅ Probe built: {probe}")
        
        # Test inference
        dummy_input = torch.randn(1, 512)  # EEGPT embedding size
        with torch.no_grad():
            output = probe(dummy_input)
            probs = torch.softmax(output, dim=-1)
            print(f"✅ Inference successful. Output shape: {output.shape}")
            print(f"   Probabilities: {probs.numpy()}")
            
        print("\n🎉 All tests passed! Ready for integration.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)