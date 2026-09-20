"""Tiga compiler, minimal Tensor runtime, and semantic autograd.

Ordinary Torch applications use Torch tensors, installed separately by the user.
Torch is optional: lazy imports keep the native compiler, Tensor/runtime and
MessagePassing/autograd paths usable without it.
"""

from __future__ import annotations

from importlib import import_module

from . import autograd, compiler, control, math, nn, runtime, stencil, visualize
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
from .runtime.memory import execution
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
    "MeanReducer": (".reducer", "MeanReducer"),
    "ProductReducer": (".reducer", "ProductReducer"),
    "Reducer": (".reducer", "Reducer"),
    "ReducerCall": (".reducer", "ReducerCall"),
    "SumReducer": (".reducer", "SumReducer"),
    "mean": (".reducer", "mean"),
    "online_softmax": (".reducer", "online_softmax"),
    "prod": (".reducer", "prod"),
    "sum": (".reducer", "sum"),
}


def load(path, *, device="cpu"):
    """Lazily open a Tensor snapshot file or a versioned Graph directory.

    Tensor snapshots restore values, not autograd history. Payload validation
    occurs on attachment; native allocation is deferred until first use.
    """
    from pathlib import Path
    if Path(path).is_file():
        from .tensor.spill import load_tensor
        return load_tensor(path, device=device)
    from .graph import Graph

    return Graph.open(path, device=device)


def save(graph, path, *, fields=None, overwrite=False) -> None:
    """Persist a Tensor value snapshot or a static CSR Graph.

    Tensor snapshots refuse existing paths unless ``overwrite=True``. They
    do not evict the value or serialize its autograd graph. Graph overwriting
    is unsupported; use a new directory.

    ``fields`` optionally stores node/edge payloads inside the ``.gfg`` as
    fixed-row binary files — ``{"src": {...}, "dst": {...}, "edge": {...}}``.
    ``tg.load(path).fields(role)`` exposes them as disk-backed shell Tensors
    for bounded-memory paged execution.
    """
    if isinstance(graph, Tensor):
        if fields is not None:
            raise TypeError("fields applies only to Graph snapshots")
        graph.save(path, overwrite=overwrite)
        return
    if overwrite:
        raise ValueError("Graph overwrite is not supported; save to a new directory")
    from .graph import save_graph

    save_graph(graph, path, fields=fields)


def from_disk(name: str) -> Tensor:
    """Attach a named on-disk spill as a lazily-loaded Tensor.

    The counterpart of ``tensor.disk(name=name)``: another process (or a
    later run) reads the payload from the shared spill directory on first
    use. Raises ``FileNotFoundError`` for unknown names.
    """
    from .tensor.spill import open_spill

    return open_spill(name)


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module 'tiga' has no attribute {name!r}") from error
    try:
        value = getattr(import_module(module_name, __name__), attribute)
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise ModuleNotFoundError(
                f"tiga.{name} requires the Torch adapter; "
                "install tiga-lang[torch]. Tensor/runtime/autograd "
                "remain available without Torch."
            ) from error
        raise
    globals()[name] = value
    return value


__all__ = [
    "stencil",
    "execution",
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
    "MeanReducer",
    "MessagePassing",
    "OnlineSoftmaxItem",
    "OnlineSoftmaxReducer",
    "ProductReducer",
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
    "from_disk",
    "load",
    "math",
    "mean",
    "nn",
    "ones_like",
    "online_softmax",
    "prod",
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
