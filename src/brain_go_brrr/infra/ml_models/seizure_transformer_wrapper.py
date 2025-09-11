"""SeizureTransformer wrapper with safe imports and overlap aggregation.

This wrapper expects either:
- An externally provided model instance (dependency injection), or
- A pip-installable package exposing `build_seizure_transformer(n_channels: int)`.

Weights loading is optional and performed via `torch.load` with `weights_only=True`
when possible. Inputs are Volts in `(C, T)` at 256 Hz by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn


class SeizureTransformerWrapper:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        build_fn: Optional[Callable[[int], nn.Module]] = None,
        weights_path: Optional[Path | str] = None,
        n_channels: int = 19,
        fs: int = 256,
        window_sec: float = 60.0,
        stride_sec: float = 30.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.fs = fs
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.n_channels = n_channels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None:
            self.model = model
        else:
            if build_fn is None:
                # Try to import a known entry point; otherwise raise with instructions
                try:  # pragma: no cover - optional integration
                    from seizure_transformer import build_seizure_transformer  # type: ignore

                    build_fn = build_seizure_transformer  # type: ignore[assignment]
                except Exception as e:  # pragma: no cover
                    raise ImportError(
                        "Could not import SeizureTransformer. Either pass a `model` or `build_fn`, "
                        "or install the upstream package exposing `build_seizure_transformer`."
                    ) from e
            assert build_fn is not None
            self.model = build_fn(self.n_channels)

        if weights_path is not None:
            p = Path(weights_path)
            if p.exists():
                ckpt = torch.load(p, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
                state_dict = ckpt.get("model", ckpt)
                self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, eeg: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Return per-sample seizure probabilities for a single recording.

        Args:
            eeg: Array of shape (C, T) in Volts at `fs` Hz.
        Returns:
            probs: Array of shape (T,) with values in [0, 1].
        """
        assert eeg.ndim == 2, "expected (C, T)"
        c, t = int(eeg.shape[0]), int(eeg.shape[1])
        if c != self.n_channels:
            raise ValueError(f"expected {self.n_channels} channels, got {c}")

        win = int(round(self.window_sec * self.fs))
        stride = int(round(self.stride_sec * self.fs))
        if t < win:
            # Pad to window length
            pad = win - t
            eeg = np.pad(eeg, ((0, 0), (0, pad)), mode="edge")
            t = win

        probs = np.zeros(t, dtype=np.float32)
        counts = np.zeros(t, dtype=np.int32)

        # Sliding windows with overlap aggregation by averaging
        starts = range(0, t - win + 1, stride)
        for s in starts:
            e = s + win
            x = torch.from_numpy(eeg[:, s:e]).unsqueeze(0).to(self.device)
            if torch.cuda.is_available():
                amp = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            else:
                import contextlib

                amp = contextlib.nullcontext()
            with amp:
                logits = self.model(x)  # shape (B, T') or (B, 1); depends on model
                out = torch.sigmoid(logits)
                # Map to window samples: if scalar, broadcast; if temporal, mean over time
                if out.ndim == 2 and out.shape[-1] > 1:
                    p = out.mean(dim=-1).squeeze(0).item()
                else:
                    p = out.squeeze().item()
            probs[s:e] += np.float32(p)
            counts[s:e] += 1

        counts = np.maximum(counts, 1)
        probs = probs / counts
        return probs[: int(eeg.shape[1])]


__all__ = ["SeizureTransformerWrapper"]

