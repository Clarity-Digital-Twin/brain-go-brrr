"""Config module shim for backward compatibility.

DEPRECATED: Use brain_go_brrr.application.config instead.
This shim will be removed in v2.0.0.
"""

from brain_go_brrr.utils.deprecated_redirect import redirect

# Silent redirect from old to new location
redirect(
    old="brain_go_brrr.config",
    new="brain_go_brrr.application.config",
    globals_dict=globals(),
    warn_on_import=False,  # Don't warn - too noisy
)
