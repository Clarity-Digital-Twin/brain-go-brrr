# DEPRECATED: use brain_go_brrr.data.edf_loader instead. Removed in v2.0.0.
from brain_go_brrr.utils.deprecated_redirect import redirect

redirect(
    old="brain_go_brrr.core.edf_loader",
    new="brain_go_brrr.data.edf_loader",
    globals_dict=globals(),
    warn_on_import=False,
)
