"""Compatibility shim for moved features module.

DEPRECATED: Use brain_go_brrr.preprocessing.features instead.
This shim will be removed in a future release.
"""

import sys
import warnings

warnings.warn(
    "brain_go_brrr.core.features has moved to brain_go_brrr.preprocessing.features",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new location
from brain_go_brrr.preprocessing.features import *  # noqa: F403, E402

# Also make submodules available for tests that import from .extractor
from brain_go_brrr.preprocessing.features import (  # noqa: E402
    __all__,  # noqa: F401
    extractor,
)

sys.modules['brain_go_brrr.core.features.extractor'] = extractor
