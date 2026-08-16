"""GraphForge compiler, minimal Tensor runtime, and semantic autograd.

Torch integration is lazy and optional. The compiler tools and native CPU
Tensor/runtime surface can be imported without Torch; accessing Graph,
MessagePassing or the bootstrap Torch adapter loads it on demand.
"""

from __future__ import annotations

from importlib import import_module

from . import autograd, compiler, control, math, runtime, visualize
from ._version import __version__
from .control import repeat, while_loop
from .distributed import (
    ByDestination,
    DeviceMesh,
    GraphPlacement,
    HaloMap,
    collective_halo_maps,
    derive_halo_map,
    exchange_halo,
)
from .program import GraphProgram, ProgramValue, program
from .runtime import Device, DeviceType
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
    "jit": (".jit", "jit"),
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
    "AnalysisFinding",
    "ByDestination",
    "CompiledVariant",
    "DType",
    "Device",
    "DeviceMesh",
    "DeviceType",
    "Graph",
    "GraphPlacement",
    "GraphProgram",
    "GraphSchema",
    "HaloMap",
    "Kernel",
    "MachineSchedule",
    "MessagePassing",
    "OnlineSoftmaxItem",
    "OnlineSoftmaxReducer",
    "ProgramValue",
    "RadiusGraph",
    "Reducer",
    "ReducerCall",
    "SumReducer",
    "Tensor",
    "__version__",
    "autograd",
    "bool",
    "collective_halo_maps",
    "compiler",
    "complex64",
    "complex128",
    "control",
    "derive_halo_map",
    "empty",
    "exchange_halo",
    "float16",
    "float32",
    "float64",
    "from_torch",
    "int32",
    "int64",
    "jit",
    "load",
    "math",
    "ones_like",
    "online_softmax",
    "program",
    "repeat",
    "runtime",
    "save",
    "sum",
    "tensor",
    "visualize",
    "while_loop",
    "zeros_like",
]
