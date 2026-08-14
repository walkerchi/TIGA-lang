"""GraphForge compiler, minimal Tensor runtime, and semantic autograd.

Torch integration is lazy and optional. The compiler tools and native CPU
Tensor/runtime surface can be imported without Torch; accessing Graph,
MessagePassing or the bootstrap Torch adapter loads it on demand.
"""

from __future__ import annotations

from importlib import import_module

from ._version import __version__
from . import autograd, compiler, math, runtime, visualize
from .distributed import (
    ByDestination, DeviceMesh, GraphPlacement, HaloMap, derive_halo_map,
    collective_halo_maps, exchange_halo,
)
from .runtime import Device, DeviceType
from .program import GraphProgram, ProgramValue, program
from .tensor import (
    DType,
    Tensor,
    bool,
    complex64,
    complex128,
    empty,
    float16,
    float32,
    float64,
    from_torch,
    int32,
    int64,
    ones_like,
    tensor,
    zeros_like,
)


_LAZY_EXPORTS = {
    "AnalysisFinding": (".kernel", "AnalysisFinding"),
    "CompiledVariant": (".kernel", "CompiledVariant"),
    "Kernel": (".kernel", "Kernel"),
    "MachineSchedule": (".kernel", "MachineSchedule"),
    "Graph": (".graph", "Graph"),
    "GraphSchema": (".graph", "GraphSchema"),
    "MessagePassing": (".message_passing", "MessagePassing"),
    "RadiusGraph": (".graph", "RadiusGraph"),
    "OnlineSoftmaxItem": (".reducer", "OnlineSoftmaxItem"),
    "OnlineSoftmaxReducer": (".reducer", "OnlineSoftmaxReducer"),
    "Reducer": (".reducer", "Reducer"),
    "ReducerCall": (".reducer", "ReducerCall"),
    "SumReducer": (".reducer", "SumReducer"),
    "online_softmax": (".reducer", "online_softmax"),
    "sum": (".reducer", "sum"),
}


def load(path, *, device="cpu"):
    """Open a versioned persistent Graph without eager topology loading."""
    from .graph import Graph

    return Graph.open(path, device=device)


def save(graph, path) -> None:
    """Persist a static CSR Graph in the versioned GraphForge format."""
    from .graph import save_graph

    save_graph(graph, path)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module 'graphforge' has no attribute {name!r}") from error
    try:
        value = getattr(import_module(module_name, __name__), attribute)
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ModuleNotFoundError(
                f"graphforge.{name} currently uses the optional Torch adapter; "
                "install graphforge-compiler[torch]. Tensor/runtime/autograd "
                "remain available without Torch."
            ) from error
        raise
    globals()[name] = value
    return value


__all__ = [
    "__version__",
    "AnalysisFinding",
    "ByDestination",
    "CompiledVariant",
    "DType",
    "Device",
    "DeviceType",
    "DeviceMesh",
    "Graph",
    "GraphPlacement",
    "HaloMap",
    "GraphSchema",
    "GraphProgram",
    "Kernel",
    "MachineSchedule",
    "MessagePassing",
    "OnlineSoftmaxItem",
    "OnlineSoftmaxReducer",
    "RadiusGraph",
    "ProgramValue",
    "Reducer",
    "ReducerCall",
    "SumReducer",
    "Tensor",
    "autograd",
    "bool",
    "complex64",
    "complex128",
    "compiler",
    "collective_halo_maps",
    "derive_halo_map",
    "empty",
    "exchange_halo",
    "float16",
    "float32",
    "float64",
    "from_torch",
    "int32",
    "int64",
    "load",
    "math",
    "ones_like",
    "online_softmax",
    "program",
    "runtime",
    "visualize",
    "save",
    "sum",
    "tensor",
    "zeros_like",
]
