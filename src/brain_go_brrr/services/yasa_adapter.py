# DEPRECATED: use brain_go_brrr.infra.external.yasa_adapter. Removed in v2.0.0.
# Lazy redirect to avoid import-time side effects

def __getattr__(name):
    """Lazy redirect to new location."""
    import importlib
    mod = importlib.import_module("brain_go_brrr.infra.external.yasa_adapter")
    return getattr(mod, name)
