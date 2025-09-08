"""Protocols package.

Protocol Usage Guidelines:

1. Add @runtime_checkable ONLY when:
   - Tests or runtime code use isinstance(obj, ProtocolType)
   - You need runtime type checking of a Protocol

2. Current runtime-checkable protocols:
   - LoggerPort (used in isinstance checks)

3. Protocols without @runtime_checkable:
   - CachePort, ModelPort, etc. (structural typing only)

Note: If you change runtime-checkability on protocols and see odd behavior,
clear __pycache__ to avoid stale caches during local development.
"""
