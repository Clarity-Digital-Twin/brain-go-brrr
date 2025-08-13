"""Compatibility shim for moved edf_validator module.

DEPRECATED: Use brain_go_brrr.data.edf_validator instead.
This shim will be removed in version 2.0.0.
"""

from brain_go_brrr.utils.deprecated_redirect import redirect

# Clean redirect to new location
redirect(
    __name__,
    "brain_go_brrr.data.edf_validator",
    removal_version="2.0.0"
)

# Re-export for compatibility
from brain_go_brrr.data.edf_validator import *  # noqa: F403, E402
