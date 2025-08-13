# DEPRECATED: use brain_go_brrr.config.abnormality_config instead. Removed in v2.0.0.
from brain_go_brrr.utils.deprecated_redirect import redirect
redirect(
    old="brain_go_brrr.core.abnormality_config",
    new="brain_go_brrr.config.abnormality_config",
    globals_dict=globals(),
    warn_on_import=False,
)