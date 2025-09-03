"""Unified probe factory for consolidating duplicate probe implementations.

Part of P2 technical debt cleanup - consolidates TwoLayerProbe and EEGPTProbe.
"""

from typing import Any, Literal

import torch.nn as nn

from brain_go_brrr.infra.ml_models.linear_probe import LinearProbeHead, TwoLayerProbe


class ProbeFactory:
    """Factory for creating probe models.

    This consolidates the duplicate probe implementations:
    - TwoLayerProbe from linear_probe.py
    - EEGPTProbe's two_layer mode from eegpt_probe_unified.py

    The factory approach ensures backward compatibility while providing
    a single interface for probe creation.
    """

    @staticmethod
    def create(
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        architecture: Literal["linear", "two_layer"] = "two_layer",
        dropout: float = 0.0,
        pool: Literal["mean", "max", "cls"] = "mean",
        **kwargs: Any,  # noqa: ARG004 - for future extensions
    ) -> nn.Module:
        """Create a probe with the specified architecture.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes
            architecture: "linear" for single layer, "two_layer" for MLP
            dropout: Dropout probability
            pool: Pooling strategy for 3D inputs
            **kwargs: Additional arguments for future extensions

        Returns:
            Probe module

        Examples:
            >>> # Create two-layer probe (default)
            >>> probe = ProbeFactory.create(2048, 256, 2)
            >>>
            >>> # Create linear probe
            >>> probe = ProbeFactory.create(2048, 256, 5, architecture="linear")
        """
        if architecture == "two_layer":
            # Use existing TwoLayerProbe implementation
            return TwoLayerProbe(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=dropout,
                pool=pool,
            )
        elif architecture == "linear":
            # Use LinearProbeHead for single-layer probe
            return LinearProbeHead(
                input_dim=input_dim,
                num_classes=output_dim,  # LinearProbeHead uses num_classes
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown architecture: {architecture}. Supported: 'linear', 'two_layer'"
            )

    @classmethod
    def create_for_task(
        cls,
        task: str,
        input_dim: int = 2048,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a probe for a specific task with sensible defaults.

        Args:
            task: Task name ("abnormality", "sleep", "motor_imagery")
            input_dim: Input feature dimension
            **kwargs: Override default parameters

        Returns:
            Probe configured for the task

        Examples:
            >>> # Create abnormality detection probe
            >>> probe = ProbeFactory.create_for_task("abnormality")
            >>>
            >>> # Create sleep staging probe with custom dropout
            >>> probe = ProbeFactory.create_for_task("sleep", dropout=0.3)
        """
        task_configs = {
            "abnormality": {
                "hidden_dim": 256,
                "output_dim": 2,  # Normal/Abnormal
                "architecture": "two_layer",
                "dropout": 0.25,
            },
            "sleep": {
                "hidden_dim": 256,
                "output_dim": 5,  # W, N1, N2, N3, REM
                "architecture": "two_layer",
                "dropout": 0.1,
            },
            "motor_imagery": {
                "hidden_dim": 128,
                "output_dim": 4,  # Left, Right, Feet, Tongue
                "architecture": "linear",
                "dropout": 0.1,
            },
            "tuev": {
                "hidden_dim": 256,
                "output_dim": 6,  # SPSW, GPED, PLED, EYEM, ARTF, BCKG
                "architecture": "two_layer",
                "dropout": 0.5,  # Higher dropout for TUEV
            },
        }

        if task not in task_configs:
            raise ValueError(f"Unknown task: {task}. Supported: {list(task_configs.keys())}")

        config = task_configs[task].copy()
        config["input_dim"] = input_dim
        config.update(kwargs)  # Allow overrides

        return cls.create(**config)


# Backward compatibility alias
UnifiedProbe = ProbeFactory


def migrate_eegpt_probe_to_factory(eegpt_probe_state_dict: dict[str, Any]) -> dict[str, Any]:
    """Migrate EEGPTProbe state_dict to ProbeFactory format.

    Args:
        eegpt_probe_state_dict: State dict from EEGPTProbe

    Returns:
        State dict compatible with ProbeFactory.create()

    Note:
        This is for migration purposes. EEGPTProbe's probe layers
        map directly to TwoLayerProbe's net layers.
    """
    # EEGPTProbe stores probe layers as "probe.0.weight", etc.
    # TwoLayerProbe stores them as "net.0.weight", etc.
    migrated = {}
    for key, value in eegpt_probe_state_dict.items():
        if key.startswith("probe."):
            # Replace "probe" with "net" to match TwoLayerProbe
            new_key = key.replace("probe.", "net.", 1)
            migrated[new_key] = value
        elif not key.startswith("backbone."):
            # Copy non-backbone keys as-is
            migrated[key] = value

    return migrated
