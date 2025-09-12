"""Single Source of Truth for TUSZ seizure labels and interval utilities.

Exports:
- TUSZ_EPILEPTIC_CODES: allowed epileptic seizure codes
- is_seizure_label(label: str, include_pnes: bool = False) -> bool
- merge_intervals(xs: list[tuple[float,float]], gap: float = 0.0)
"""

from __future__ import annotations

# Allowed epileptic seizure codes (lowercase) from TUSZ
TUSZ_EPILEPTIC_CODES: set[str] = {
    "fnsz",  # focal non-specific seizure
    "gnsz",  # generalized non-specific seizure
    "spsz",  # simple partial seizure
    "cpsz",  # complex partial seizure
    "absz",  # absence seizure
    "tnsz",  # tonic seizure
    "tcsz",  # tonic-clonic seizure
    "gtsz",  # generalized tonic-clonic seizure
    "mysz",  # myoclonic seizure
    "unsz",  # unclassified seizure
}

# Explicitly excluded codes and non-seizure events
TUSZ_EXCLUDED_CODES: set[str] = {
    "spsw",  # spike-and-wave (not seizure event)
    "gped",  # generalized periodic epileptiform discharges
    "pled",  # periodic lateralized epileptiform discharges
    "eyem",  # eye movement
    "artf",  # artifact
    "bckg",  # background
}


def is_seizure_label(label: str, include_pnes: bool = False) -> bool:
    """Return True if `label` denotes a seizure event.

    Rules:
    - Accept TUSZ epileptic seizure codes (case-insensitive).
    - Optionally accept PNES (NESZ) when `include_pnes=True`.
    - Explicitly exclude known non-seizure codes listed above.
    - Fallback: accept textual labels containing 'seiz' (case-insensitive).
    """
    lab = (label or "").strip().lower()
    if not lab:
        return False

    # Explicit exclusions first
    if any(code in lab for code in TUSZ_EXCLUDED_CODES):
        return False

    # PNES as opt-in
    if include_pnes and "nesz" in lab:
        return True

    # Epileptic seizure codes
    if any(code in lab for code in TUSZ_EPILEPTIC_CODES):
        return True

    # Generic textual
    return "seiz" in lab


def merge_intervals(xs: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    """Merge overlapping or touching intervals.

    Args:
        xs: List of (start, end) in seconds
        gap: Allow merging if next.start <= prev.end + gap

    Returns:
        Coalesced list of non-overlapping intervals sorted by start.
    """
    if not xs:
        return []
    xs = sorted(xs)
    out = [xs[0]]
    for s, e in xs[1:]:
        ps, pe = out[-1]
        if s <= pe + gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


__all__ = [
    "TUSZ_EPILEPTIC_CODES",
    "TUSZ_EXCLUDED_CODES",
    "is_seizure_label",
    "merge_intervals",
]
