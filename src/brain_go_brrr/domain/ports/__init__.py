"""Domain ports - abstractions for infrastructure dependencies.

Following Clean Architecture, the domain defines interfaces (ports)
that infrastructure implements (adapters).
"""

# Re-export all ports so callers can do `from brain_go_brrr.domain.ports import X`
from ..protocols.logger import LoggerPort  # P1 FIX: Import from unified location
from .base import (
    AbnormalityConfigPort,
    ConfigurationPort,
    EEGModelPort,
    PreprocessorPort,
)
from .cache import AsyncCachePort, CachePort

__all__ = [
    # From base - sorted
    "AbnormalityConfigPort",
    "AsyncCachePort",
    "CachePort",
    "ConfigurationPort",
    "EEGModelPort",
    "LoggerPort",
    "PreprocessorPort",
]
