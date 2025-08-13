# DEPRECATED: use brain_go_brrr.domain.quality instead. Removed in v2.0.0.
from brain_go_brrr.utils.deprecated_redirect import redirect
redirect(
    old="brain_go_brrr.core.quality",
    new="brain_go_brrr.domain.quality",
    globals_dict=globals(),
    warn_on_import=False,
)
