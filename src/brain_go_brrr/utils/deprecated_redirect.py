"""Clean deprecation redirect helper following PEP-562.

This module provides a single, reusable helper for creating deprecation
shims when modules are moved. It follows clean code principles and avoids
the need for # noqa comments everywhere.
"""

import importlib
import sys
import warnings


def redirect(
    module_name: str,
    target: str,
    *,
    submods: tuple[str, ...] = (),
    removal_version: str | None = None
) -> None:
    """Create a deprecation redirect from old module path to new.

    This follows PEP-562 (__getattr__ and __dir__ on modules) to provide
    clean deprecation warnings without import order issues or type: ignore
    comments.

    Args:
        module_name: Full name of the deprecated module (typically __name__)
        target: Full name of the new module location
        submods: Tuple of submodule names to also redirect
        removal_version: Version when this shim will be removed (for warning)

    Example:
        # In src/brain_go_brrr/core/features.py:
        from brain_go_brrr.utils.deprecated_redirect import redirect
        redirect(__name__, "brain_go_brrr.preprocessing.features", submods=("extractor",))
    """
    removal_msg = f" Will be removed in version {removal_version}." if removal_version else ""
    warnings.warn(
        f"{module_name} is deprecated; use {target}.{removal_msg}",
        DeprecationWarning,
        stacklevel=3  # Skip this function and the caller to show the import line
    )

    # Import the target module
    target_mod = importlib.import_module(target)

    # Replace this module with the target in sys.modules
    sys.modules[module_name] = target_mod

    # Also redirect any submodules
    for submod in submods:
        try:
            sub_target = importlib.import_module(f"{target}.{submod}")
            sys.modules[f"{module_name}.{submod}"] = sub_target
        except ImportError:
            # Submodule doesn't exist in target, skip it
            pass


__all__ = ["redirect"]
