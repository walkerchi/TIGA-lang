"""Optional zero-copy Torch storage adapter for :mod:`tiga.tensor`."""

from __future__ import annotations


class TorchBuffer:
    _tiga_torch_buffer = True

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

    def adjacent_difference_bounds(
        self, *, offset: int, stride: int, count: int
    ) -> tuple[int, int]:
        """Reduce a one-dimensional integer view without host materialization.

        This is a storage-provider analysis hook, not a graph or Torch op in
        the compiler core. Only the two extrema cross the device boundary.
        """
        if count < 2:
            return 0, 0
        import torch

        owner = self.tensor
        view = torch.as_strided(
            owner,
            (count,),
            (stride,),
            storage_offset=owner.storage_offset() + offset,
        )
        differences = view[1:] - view[:-1]
        extrema = torch.stack((differences.min(), differences.max())).cpu()
        return int(extrema[0]), int(extrema[1])

    def csr_source_index_span_ratio(
        self,
        *,
        row_ptr,
        offset: int,
        stride: int,
        count: int,
        num_src: int,
        num_dst: int,
        samples: int,
    ) -> float:
        """Estimate CSR source locality from a bounded device sample."""
        if count == 0:
            return 0.0
        import torch

        row_buffer = row_ptr._buffer
        if not getattr(row_buffer, "_tiga_torch_buffer", False):
            raise TypeError("row_ptr must use the same Torch storage provider")
        column_owner = self.tensor
        columns = torch.as_strided(
            column_owner,
            (count,),
            (stride,),
            storage_offset=column_owner.storage_offset() + offset,
        )
        row_owner = row_buffer.tensor
        rows = torch.as_strided(
            row_owner,
            (row_ptr.numel,),
            (row_ptr.strides[0],),
            storage_offset=row_owner.storage_offset() + row_ptr.offset,
        )
        sample_count = min(samples, count)
        slots = torch.arange(
            sample_count, dtype=torch.int64, device=columns.device)
        if sample_count > 1:
            slots = torch.div(
                slots * (count - 1), sample_count - 1,
                rounding_mode="floor")
        destinations = torch.searchsorted(
            rows[1:].contiguous(), slots, right=True)
        selected = columns.gather(0, slots).to(torch.int64)
        ratio = (selected - destinations).abs().to(torch.float64).mean()
        ratio = ratio / max(1, num_src, num_dst)
        return float(ratio.cpu())


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
    if not getattr(buffer, "_tiga_torch_buffer", False):
        raise RuntimeError("zero-copy to_torch requires Torch-owned storage")
    owner = buffer.tensor
    return torch.as_strided(
        owner,
        value.shape,
        value.strides,
        owner.storage_offset() + value.offset,
    )


def copy_to_torch(value):
    """Materialize a detached Torch copy of contiguous Tiga storage.

    The ordinary :func:`to_torch` contract remains zero-copy and therefore
    refuses Tiga-owned allocations.  This explicit bridge is used when
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
    if getattr(buffer, "_tiga_torch_buffer", False):
        return to_torch(value)
    if not isinstance(buffer, Buffer):
        raise RuntimeError("Tiga Tensor storage is not addressable")
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
