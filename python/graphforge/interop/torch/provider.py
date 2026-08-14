"""Torch-owned allocation helpers for compatibility provider executables."""

from __future__ import annotations

import sys

from ...runtime import ImmediateCompletion


def current_cuda_stream(device) -> object:
    """Wrap Torch's current CUDA stream in the GraphForge runtime ABI."""
    import torch
    from ...runtime import Stream

    current = torch.cuda.current_stream(device=torch.device(str(device)))
    return Stream.wrap_address(int(current.cuda_stream), device=device)


def cuda_destination_index(row_ptr, rows: int):
    """Materialize a CSR destination index using the optional Torch adapter."""
    import torch
    from ...tensor import from_torch

    raw = row_ptr.to_torch()
    counts = raw[1:] - raw[:-1]
    physical = torch.repeat_interleave(
        torch.arange(rows, device=raw.device, dtype=raw.dtype), counts,
    )
    torch.cuda.synchronize(raw.device)
    return from_torch(physical)


def cuda_pinned_roundtrip(source):
    """Execute a Torch-owned checkpoint spill/restore compatibility copy."""
    import torch
    from ...tensor import from_torch

    raw = source.to_torch()
    host = torch.empty_like(raw, device="cpu", pin_memory=True)
    host.copy_(raw, non_blocking=False)
    restored = host.to(raw.device, non_blocking=True)
    torch.cuda.synchronize(raw.device)
    return from_torch(restored)


class TorchCudaCompletion:
    """Completion backed by a CUDA stream, without eager event recording.

    Same-stream launches are already ordered. Cross-stream dependencies use
    ``wait_stream``, whose provider owns any event needed by Torch/CUDA. This
    avoids adding an event record to every single-kernel hot path.
    """

    def __init__(self, stream=None):
        self._stream = stream
        self._waited = False

    @property
    def ready(self):
        # Without a captured stream/event, false is the conservative answer.
        return self._waited or (
            self._stream is not None and bool(self._stream.query()))

    def wait(self):
        if self._stream is None:
            import torch
            torch.cuda.synchronize()
        else:
            self._stream.synchronize()
        self._waited = True


class TorchCudaSubmissionProvider:
    """Submit runtime bundle invocations on Torch's current CUDA stream."""

    def submit(self, invocation, arguments, wait_for, *, will_be_waited=False):
        import torch

        stream = torch.cuda.current_stream() if (wait_for or will_be_waited) else None
        for completion in wait_for:
            producer_stream = getattr(completion, "_stream", None)
            if producer_stream is None:
                completion.wait()
            elif producer_stream != stream:
                stream.wait_stream(producer_stream)
        executable = invocation.executable
        launch = getattr(executable, "launch", None)
        if launch is None:
            launch = executable
        launch(**arguments)
        return TorchCudaCompletion(stream)

    def prepare(self, invocation, argument_slots):
        return _PreparedTorchCudaInvocation(invocation.executable, argument_slots)


class _PreparedTorchCudaInvocation:
    def __init__(self, executable, argument_slots):
        prepare = getattr(executable, "prepare_submission", None)
        self._prepared_launch = (
            prepare(argument_slots) if prepare is not None else None)
        self._launch = getattr(executable, "launch", executable)
        self._argument_slots = argument_slots

    def submit(self, resources, wait_for, *, will_be_waited=False):
        import torch

        stream = torch.cuda.current_stream() if (wait_for or will_be_waited) else None
        for completion in wait_for:
            producer_stream = getattr(completion, "_stream", None)
            if producer_stream is None:
                completion.wait()
            elif producer_stream != stream:
                stream.wait_stream(producer_stream)
        if self._prepared_launch is not None:
            self._prepared_launch(resources)
        else:
            self._launch(**{
                parameter: resources[slot]
                for parameter, slot in self._argument_slots
            })
        return TorchCudaCompletion(stream)

    def launch_same_stream(self, resources):
        """Launch an already prepared dependency-free invocation directly.

        A caller may use this only after proving every invocation is submitted
        to the current CUDA stream and the surrounding bundle contains no
        communication/cross-stream dependency.  The stream then is the event
        chain, so allocating Completion wrappers adds no semantics.
        """
        if self._prepared_launch is not None:
            self._prepared_launch(resources)
        else:
            self._launch(**{
                parameter: resources[slot]
                for parameter, slot in self._argument_slots
            })


class ImmediateSubmissionProvider:
    """Small synchronous provider retained for non-CUDA compatibility paths."""

    def submit(self, invocation, arguments, wait_for, *, will_be_waited=False):
        del will_be_waited
        for completion in wait_for:
            completion.wait()
        launch = getattr(invocation.executable, "launch", invocation.executable)
        launch(**arguments)
        return ImmediateCompletion()

    def prepare(self, invocation, argument_slots):
        return _PreparedImmediateInvocation(invocation.executable, argument_slots)


class _PreparedImmediateInvocation:
    def __init__(self, executable, argument_slots):
        self._launch = getattr(executable, "launch", executable)
        self._argument_slots = argument_slots

    def submit(self, resources, wait_for, *, will_be_waited=False):
        del will_be_waited
        for completion in wait_for:
            completion.wait()
        self._launch(**{
            parameter: resources[slot]
            for parameter, slot in self._argument_slots
        })
        return ImmediateCompletion()


def reusable_empty_like(current, reference):
    import torch

    # At this call site the owning plan and this function argument account for
    # two strong references; getrefcount() contributes the third temporary
    # reference.  Four or more therefore proves that a caller still owns a
    # previously returned output and that overwriting it would be observable.
    if current is None or sys.getrefcount(current) > 3:
        return torch.empty_like(reference)
    return current


def reusable_dense_output(current, lhs, *, lanes: int, rows: int, width: int):
    import torch

    # Dense plans return a permuted view rather than ``current`` itself.
    # Holding that view increments TensorImpl's storage ownership but need not
    # add a Python reference to the base tensor, so check both ownership layers.
    has_live_view = (
        current is not None
        and hasattr(current, "_use_count")
        and current._use_count() > 1
    )
    if current is None or sys.getrefcount(current) > 3 or has_live_view:
        current = torch.empty(
            (lanes, rows, width), device=lhs.device, dtype=lhs.dtype
        )
    return current, current.permute(1, 0, 2)


def reusable_vector_output(current, reference, *, rows: int):
    import torch

    if (
        current is None
        or sys.getrefcount(current) > 3
        or current.shape != (rows,)
        or current.device != reference.device
        or current.dtype != reference.dtype
    ):
        current = torch.empty((rows,), device=reference.device,
                              dtype=reference.dtype)
    return current


__all__ = [
    "ImmediateSubmissionProvider",
    "TorchCudaCompletion",
    "TorchCudaSubmissionProvider",
    "cuda_destination_index",
    "cuda_pinned_roundtrip",
    "current_cuda_stream",
    "reusable_dense_output",
    "reusable_empty_like",
    "reusable_vector_output",
]
