"""Job management for asynchronous EEG analysis tasks."""

# Lazy imports to avoid circular dependencies
__all__ = ["ThreadSafeJobStore", "get_job_store"]

def __getattr__(name):
    if name == "ThreadSafeJobStore":
        from brain_go_brrr.application.jobs.store import ThreadSafeJobStore
        return ThreadSafeJobStore
    elif name == "get_job_store":
        from brain_go_brrr.application.jobs.store import get_job_store
        return get_job_store
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
