"""Compatibility shim for moved exceptions module.

DEPRECATED: Use brain_go_brrr.domain.exceptions instead.
This shim will be removed in version 2.0.0.
"""

from brain_go_brrr.utils.deprecated_redirect import redirect

# Clean redirect to new location
redirect(
    __name__,
    "brain_go_brrr.domain.exceptions",
    removal_version="2.0.0"
)

# Re-export for compatibility
from brain_go_brrr.domain.exceptions import *  # noqa: F403, E402