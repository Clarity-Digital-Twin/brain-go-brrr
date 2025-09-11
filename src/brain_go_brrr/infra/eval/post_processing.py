"""Post-processing for temporal seizure probabilities.

Implements dual-threshold hysteresis, gap merging, and minimum duration filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt


@dataclass
class AdvancedPostProcessor:
    hysteresis: tuple[float, float] = (0.3, 0.7)  # (low, high)
    merge_gap_sec: float = 2.0
    min_duration_sec: float = 1.0
    fs: int = 256

    def _hysteresis_events(self, probs: npt.NDArray[np.floating]) -> list[tuple[int, int, float]]:
        low, high = self.hysteresis
        events: list[tuple[int, int, float]] = []
        n = probs.shape[0]
        i = 0
        while i < n:
            if probs[i] >= high:
                start = i
                max_p = float(probs[i])
                i += 1
                while i < n and probs[i] >= low:
                    if probs[i] > max_p:
                        max_p = float(probs[i])
                    i += 1
                end = i
                events.append((start, end, max_p))
            else:
                i += 1
        return events

    def _merge(self, events: Sequence[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        if not events:
            return []
        gap = int(round(self.merge_gap_sec * self.fs))
        merged: list[tuple[int, int, float]] = []
        s0, e0, c0 = events[0]
        for s, e, c in events[1:]:
            if s - e0 <= gap:
                e0 = e
                c0 = max(c0, c)
            else:
                merged.append((s0, e0, c0))
                s0, e0, c0 = s, e, c
        merged.append((s0, e0, c0))
        return merged

    def _filter_min_dur(self, events: Sequence[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
        min_len = int(round(self.min_duration_sec * self.fs))
        return [(s, e, c) for s, e, c in events if (e - s) >= min_len]

    def apply(self, probs: npt.NDArray[np.floating]) -> list[tuple[float, float, float]]:
        """Convert probabilities to (start_sec, end_sec, confidence) events."""
        events_idx = self._hysteresis_events(probs)
        events_idx = self._merge(events_idx)
        events_idx = self._filter_min_dur(events_idx)
        return [(s / self.fs, e / self.fs, c) for s, e, c in events_idx]


__all__ = ["AdvancedPostProcessor"]

