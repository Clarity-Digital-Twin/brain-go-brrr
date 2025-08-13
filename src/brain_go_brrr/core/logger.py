"""Compatibility shim for moved logger module.

DEPRECATED: Use brain_go_brrr.infra.logger instead.
This shim will be removed in version 2.0.0.
"""

from brain_go_brrr.utils.deprecated_redirect import redirect

# Clean redirect to new location
redirect(
    __name__,
    "brain_go_brrr.infra.logger",
    removal_version="2.0.0"
)

# Re-export for compatibility
from brain_go_brrr.infra.logger import *  # noqa: F403, E402