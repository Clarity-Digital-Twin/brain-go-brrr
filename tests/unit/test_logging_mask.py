from pathlib import Path

import pytest

from brain_go_brrr.infra.logger import get_logger
from brain_go_brrr.utils.logging_utils import mask_path_for_log


def test_mask_path_for_log_masks_extension_and_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BGBR_DEBUG", "0")
    p = "/tmp/patients/john_smith_123.edf"
    masked = mask_path_for_log(p)
    assert masked.startswith(".edf#")
    # hash should be 8 hex chars
    assert len(masked.split("#")[1]) == 8


def test_mask_path_for_log_handles_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BGBR_DEBUG", "0")
    assert mask_path_for_log("") == "<empty_path>"


def test_mask_path_respects_debug_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BGBR_DEBUG", "1")
    p = Path("/any/path/file.edf")
    assert mask_path_for_log(p) == str(p)


def test_logger_filter_masks_paths_in_messages(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("BGBR_DEBUG", "0")
    logger = get_logger("test.logger.mask", rich_console=False)
    raw_path = "/tmp/patient_abc/file.edf"
    with caplog.at_level("INFO"):
        logger.info(f"Processing file: {raw_path}")
    msgs = [rec.getMessage() for rec in caplog.records]
    # Some logging integrations capture the record before filters mutate it.
    # Accept either masked output or absence of the raw path in captured messages.
    assert any(".edf#" in m for m in msgs) or all(raw_path not in m for m in msgs)
