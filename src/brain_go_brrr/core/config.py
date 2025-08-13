"""Compatibility shim for moved config module.

DEPRECATED: Use brain_go_brrr.config instead.
This shim will be removed in version 2.0.0.
"""

from brain_go_brrr.utils.deprecated_redirect import redirect

# Clean redirect to new location
redirect(
    __name__,
    "brain_go_brrr.config.base",
    removal_version="2.0.0"
)

# Re-export for compatibility
from brain_go_brrr.config.base import *  # noqa: F403, E402