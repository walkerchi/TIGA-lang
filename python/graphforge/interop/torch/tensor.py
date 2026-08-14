"""Optional zero-copy Torch storage adapter for :mod:`graphforge.tensor`."""

from __future__ import annotations


class TorchBuffer:
    _graphforge_torch_buffer = True

    def __init__(self, value: object) -> None:
        from ...runtime import Device

        self.tensor = value
        self.device = Device.parse(str(value.device))  # type: ignore[attr-defined]

    @property
    def nbytes(self) -> int:
        value = self.tensor
        storage = value.untyped_storage()  # type: ignore[attr-defined]
        consumed = value.storage_offset() * value.element_size()  # type: ignore[attr-defined]
        return storage.nbytes() - consumed

    @property
    def address(self) -> int:
        return int(self.tensor.data_ptr())  # type: ignore[attr-defined]


def torch_dtype(dtype):
    import torch

    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }[dtype.name]


def allocate_buffer(shape, dtype, device) -> TorchBuffer:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "CUDA Tensor storage requires the optional Torch adapter"
        ) from error
    return TorchBuffer(
        torch.empty(shape, dtype=torch_dtype(dtype), device=str(device))
    )


def from_torch(value: object, *, requires_grad: bool | None = None):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("from_torch requires the optional Torch adapter") from error
    from ...tensor.core import (
        Tensor,
        bool as gf_bool,
        complex64,
        complex128,
        float16,
        float32,
        float64,
        int32,
        int64,
    )

    if not isinstance(value, torch.Tensor):
        raise TypeError("from_torch expects a torch.Tensor")
    dtype_by_torch = {
        torch.float16: float16,
        torch.float32: float32,
        torch.float64: float64,
        torch.int32: int32,
        torch.int64: int64,
        torch.bool: gf_bool,
        torch.complex64: complex64,
        torch.complex128: complex128,
    }
    try:
        dtype = dtype_by_torch[value.dtype]
    except KeyError as error:
        raise TypeError(f"unsupported Torch dtype {value.dtype}") from error
    requested_grad = value.requires_grad if requires_grad is None else requires_grad
    return Tensor(
        tuple(value.shape),
        dtype=dtype,
        device=str(value.device),
        buffer=TorchBuffer(value),
        strides=tuple(value.stride()),
        requires_grad=requested_grad,
    )


def to_torch(value):
    import torch

    value.realize()
    buffer = value._buffer
    if not getattr(buffer, "_graphforge_torch_buffer", False):
        raise RuntimeError("zero-copy to_torch requires Torch-owned storage")
    owner = buffer.tensor
    return torch.as_strided(
        owner,
        value.shape,
        value.strides,
        owner.storage_offset() + value.offset,
    )


def copy_to_torch(value):
    """Materialize a detached Torch copy of contiguous GraphForge storage.

    The ordinary :func:`to_torch` contract remains zero-copy and therefore
    refuses GraphForge-owned allocations.  This explicit bridge is used when
    a native/distributed execution necessarily produced runtime-owned memory.
    It performs one native D2D/H2H copy and never stages CUDA data through the
    host.
    """
    import torch

    from ...runtime import Buffer, Stream

    value.realize()
    if not value.is_contiguous:
        raise RuntimeError("copy_to_torch currently requires contiguous storage")
    buffer = value._buffer
    if getattr(buffer, "_graphforge_torch_buffer", False):
        return to_torch(value)
    if not isinstance(buffer, Buffer):
        raise RuntimeError("GraphForge Tensor storage is not addressable")
    result = torch.empty(
        value.shape, dtype=torch_dtype(value.dtype), device=str(value.device)
    )
    destination = Buffer.wrap_address(
        int(result.data_ptr()), result.numel() * result.element_size(),
        device=value.device, owner=result,
    )
    stream = Stream(value.device)
    try:
        completion = buffer.copy_to(
            destination,
            stream=stream,
            source_offset=value.offset * value.dtype.itemsize,
            bytes=value.nbytes,
        )
        completion.wait()
    finally:
        stream.close()
        destination.close()
    return result


def read_flat(value) -> list[object]:
    return to_torch(value).detach().cpu().contiguous().reshape(-1).tolist()


def write_flat(value, values) -> None:
    import torch

    physical = torch.tensor(
        list(values), dtype=torch_dtype(value.dtype), device=str(value.device)
    ).reshape(value.shape)
    to_torch(value).copy_(physical)


def synchronize(device) -> None:
    import torch

    torch.cuda.synchronize(device.ordinal)


__all__ = [
    "TorchBuffer",
    "allocate_buffer",
    "from_torch",
    "read_flat",
    "synchronize",
    "to_torch",
    "torch_dtype",
    "write_flat",
]
