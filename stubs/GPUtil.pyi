"""Type stubs for GPUtil."""

def getGPUs() -> list[GPU]: ...  # noqa: N802
def getFirstAvailable(  # noqa: N802
    order: str = ...,
    maxMemory: float = ...,  # noqa: N803
    attempts: int = ...,
    interval: float = ...,
    verbose: bool = ...,
) -> int | None: ...
def getAvailable(  # noqa: N802
    order: str = ...,
    limit: int = ...,
    maxMemory: float = ...,  # noqa: N803
    maxLoad: float = ...,  # noqa: N803
) -> list[int]: ...
def showUtilization(  # noqa: N802
    all: bool = ...,
    attrList: list[str] | None = ...,  # noqa: N803
    useOldCode: bool = ...,  # noqa: N803
) -> None: ...

class GPU:
    id: int
    name: str
    load: float
    memoryFree: float  # noqa: N815
    memoryUsed: float  # noqa: N815
    memoryTotal: float  # noqa: N815
    temperature: float
    uuid: str
    def __init__(self) -> None: ...
