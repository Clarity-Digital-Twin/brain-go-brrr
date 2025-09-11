"""Adapter surface for NEDC clinical evaluation.

This module provides a minimal, stable interface. By default it computes
basic proxy metrics; for official reporting, integrate `nedc_eeg_eval` and
replace the internal implementations accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


def _events_overlap(
    e1: Tuple[float, float], e2: Tuple[float, float], min_jaccard: float = 0.5
) -> bool:
    s1, e1_ = e1
    s2, e2_ = e2
    inter = max(0.0, min(e1_, e2_) - max(s1, s2))
    union = max(e1_, e2_) - min(s1, s2)
    j = inter / union if union > 0 else 0.0
    return j >= min_jaccard


@dataclass
class NEDCClinicalEvaluator:
    sensitivity_levels: tuple[float, ...] = (0.95,)

    def compute_all_metrics(
        self,
        predictions: Iterable[Tuple[float, float]],
        ground_truth: Iterable[Tuple[float, float]],
        duration_hours: float,
    ) -> Dict[str, float]:
        """Compute proxy metrics approximating clinical KPIs.

        This is NOT a substitute for `nedc_eeg_eval`. For publication-grade
        metrics, invoke NEDC and parse its outputs. This adapter exists to keep
        our code hermetic and testable.
        """
        preds = list(predictions)
        refs = list(ground_truth)

        # Sensitivity: fraction of ref events overlapped by any prediction
        tp = 0
        for r in refs:
            if any(_events_overlap(r, p) for p in preds):
                tp += 1
        sensitivity = tp / len(refs) if refs else 0.0

        # False alarms: predicted events with no overlap, normalized to 24h
        fa = 0
        for p in preds:
            if not any(_events_overlap(p, r) for r in refs):
                fa += 1
        fa_per_24h = (fa / max(duration_hours, 1e-6)) * 24.0

        # Simple TAES-like F1 (proxy): precision/recall from event overlap
        fp = fa
        fn = len(refs) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = sensitivity
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "sensitivity": sensitivity,
            "fa_24h": fa_per_24h,
            "taes_f1": f1,
        }


__all__ = ["NEDCClinicalEvaluator"]
