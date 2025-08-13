# DEPRECATED: use brain_go_brrr.application.ports.cache instead. Removed in v2.0.0.
from brain_go_brrr.utils.deprecated_redirect import redirect

redirect(
    old="brain_go_brrr.core.cache_port",
    new="brain_go_brrr.application.ports.cache",
    globals_dict=globals(),
    warn_on_import=False,
)
