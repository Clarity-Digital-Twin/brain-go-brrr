# DEPRECATED: use brain_go_brrr.preprocessing.window_extractor instead. Removed in v2.0.0.
from brain_go_brrr.utils.deprecated_redirect import redirect

redirect(
    old="brain_go_brrr.core.window_extractor",
    new="brain_go_brrr.preprocessing.window_extractor",
    globals_dict=globals(),
    warn_on_import=False,
)
