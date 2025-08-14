"""Back-compat shim: re-export the clean extractor without emitting warnings.

pytest treats DeprecationWarning as error in this repo, so keep this silent.
"""

from .extractor_clean import *  # noqa: F403
