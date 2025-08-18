"""Backwards compatibility module for cache_port.

DEPRECATED: Import from brain_go_brrr.infra.cache instead.
"""

import warnings

warnings.warn(
    "brain_go_brrr.core.cache_port is deprecated. "
    "Import from brain_go_brrr.infra.cache instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
from brain_go_brrr.infra.cache import RedisCacheProtocol as CachePort

# Create AsyncCachePort as alias for now (async not yet implemented)
AsyncCachePort = CachePort

__all__ = ["CachePort", "AsyncCachePort"]
