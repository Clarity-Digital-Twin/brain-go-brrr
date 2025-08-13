"""Clean deprecation redirect helper following PEP-562.

This module provides a single, reusable helper for creating deprecation
shims when modules are moved. It follows clean code principles and avoids
the need for # noqa comments everywhere.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from types import ModuleType  # noqa: TC003 - Used at runtime
from typing import Any


def redirect(
    old: str,
    new: str,
    globals_dict: dict[str, Any],
    *,
    warn_on_import: bool = True,
    message: str | None = None,
) -> ModuleType:
    """PEP-562-friendly module redirect, optionally silent on import.

    Args:
        old: Full name of the deprecated module
        new: Full name of the new module location
        globals_dict: Global dictionary of the calling module (pass globals())
        warn_on_import: Whether to warn on import (False for silent redirects)
        message: Custom deprecation message

    Returns:
        The redirected module
    """
    mod = importlib.import_module(new)
    sys.modules[old] = mod
    globals_dict.update(mod.__dict__)

    if warn_on_import:
        warnings.warn(
            message or f"{old} is deprecated; use {new}. Will be removed in version 2.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return mod


__all__ = ["redirect"]
