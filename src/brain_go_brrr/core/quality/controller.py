# Shim for backward compatibility with old tests
from brain_go_brrr.domain.quality.controller import *  # noqa: F403

# Import what might be used by tests
from brain_go_brrr.domain.quality.controller import (
    CleanQualityController as EEGQualityController,  # noqa: F401
)
