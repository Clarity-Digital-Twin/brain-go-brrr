"""Tests for AbnormalityDetectionProbe checkpoint loading formats.

These tests avoid heavy model initialization by monkeypatching the EEGPT wrapper
to return a lightweight dummy backbone with the expected interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from brain_go_brrr.application.use_cases.tasks.abnormality_detection import (
    AbnormalityDetectionProbe,
)
from brain_go_brrr.infra.ml_models.probe_factory import (
    ProbeFactory,
    migrate_eegpt_probe_to_factory,
)

if TYPE_CHECKING:
    from pathlib import Path


class DummyBackbone:
    def __init__(self) -> None:
        """Initialize dummy backbone."""
        pass

    def parameters(self):
        return []

    def eval(self) -> None:
        return None

    def extract_features(self, x: torch.Tensor, channel_names=None, summary=False):
        # Return deterministic zeros of shape (B, 4, 512)
        b = x.shape[0]
        return torch.zeros((b, 4, 512), dtype=torch.float32, device=x.device)


@pytest.fixture(autouse=True)
def patch_wrapper(monkeypatch: pytest.MonkeyPatch):
    import brain_go_brrr.infra.ml_models.eegpt_wrapper as wrapper

    monkeypatch.setattr(
        wrapper, "create_normalized_eegpt", lambda checkpoint_path=None: DummyBackbone()
    )
    yield


def _make_legacy_probe_state(head: torch.nn.Module) -> dict[str, torch.Tensor]:
    # Convert TwoLayerProbe state_dict keys from net.* to probe.* to emulate legacy
    legacy: dict[str, torch.Tensor] = {}
    for k, v in head.state_dict().items():
        new_key = k.replace("net.", "probe.", 1) if k.startswith("net.") else k
        legacy[new_key] = v.clone()
    return legacy


def test_load_head_checkpoint_supports_both_formats(tmp_path: Path) -> None:
    torch.manual_seed(0)
    # Create model and a deterministic head state
    probe = AbnormalityDetectionProbe(checkpoint_path=tmp_path / "dummy.ckpt")
    reference_head = ProbeFactory.create_for_task("abnormality", n_classes=2)

    # Save current head weights in both formats
    legacy = {"probe_state_dict": _make_legacy_probe_state(reference_head)}
    current = {"model_state_dict": reference_head.state_dict()}

    # Load legacy
    probe.load_head_checkpoint(legacy)
    out1 = probe.head(torch.randn(3, 2048))

    # Load current
    probe.load_head_checkpoint(current)
    out2 = probe.head(torch.randn(3, 2048))

    assert out1.shape == out2.shape == (3, 2)


def test_migration_roundtrip_parity() -> None:
    torch.manual_seed(123)
    head = ProbeFactory.create_for_task("abnormality", n_classes=2)
    legacy = _make_legacy_probe_state(head)
    migrated = migrate_eegpt_probe_to_factory(legacy)
    new_head = ProbeFactory.create_for_task("abnormality", n_classes=2)
    new_head.load_state_dict(migrated)

    x = torch.randn(5, 2048)
    torch.manual_seed(123)
    y1 = head(x)
    torch.manual_seed(123)
    y2 = new_head(x)

    # Parity within tolerance
    assert torch.allclose(y1, y2, atol=1e-5)
