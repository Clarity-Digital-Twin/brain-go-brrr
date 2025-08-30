"""Unit tests for EDF loader path validation and error translation.

Avoids reading real EDF by patching mne.io.read_raw_edf.
"""

from __future__ import annotations

import pytest

import brain_go_brrr.infra.data.edf_loader as l
from brain_go_brrr.domain.exceptions import EdfLoadError

try:  # pragma: no cover - environment guard
    pass  # type: ignore
except Exception:  # pragma: no cover - skip if MNE unavailable
    pytest.skip("mne not available", allow_module_level=True)


def test_validate_edf_path_errors(tmp_path):
    missing = tmp_path / "no.edf"
    with pytest.raises(FileNotFoundError):
        l.validate_edf_path(missing)

    not_file = tmp_path / "dir"
    not_file.mkdir()
    with pytest.raises(ValueError):
        l.validate_edf_path(not_file)

    wrong_ext = tmp_path / "x.txt"
    wrong_ext.write_text("hi")
    with pytest.raises(ValueError):
        l.validate_edf_path(wrong_ext)


def test_load_edf_safe_translates_errors(monkeypatch, tmp_path):
    # Create a dummy path
    p = tmp_path / "x.edf"
    p.write_bytes(b"0")

    # Patch mne.io.read_raw_edf to raise common exceptions
    import mne

    # FileNotFoundError
    monkeypatch.setattr(
        mne.io, "read_raw_edf", lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("x"))
    )
    with pytest.raises(EdfLoadError, match="not found"):
        l.load_edf_safe(p)

    # ValueError (corrupt)
    monkeypatch.setattr(
        mne.io, "read_raw_edf", lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(EdfLoadError, match="Invalid EDF"):
        l.load_edf_safe(p)

    # MemoryError
    monkeypatch.setattr(
        mne.io, "read_raw_edf", lambda *_a, **_k: (_ for _ in ()).throw(MemoryError("oom"))
    )
    with pytest.raises(EdfLoadError, match="Insufficient memory"):
        l.load_edf_safe(p)

    # OSError
    monkeypatch.setattr(
        mne.io, "read_raw_edf", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fs"))
    )
    with pytest.raises(EdfLoadError, match="File system error"):
        l.load_edf_safe(p)
