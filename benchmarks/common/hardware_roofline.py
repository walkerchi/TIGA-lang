"""Device roof calibration and timing shared across workload families."""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

import torch


@dataclass
class Roof:
    dram_bandwidth_gbs: float
    l2_bandwidth_gbs: float
    fp32_gflops: float
    dram_tensor_bytes: int
    l2_tensor_bytes: int
    matmul_size: int


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def samples_ms(
    fn, device: torch.device, repeat: int, flush: torch.Tensor | None
) -> list[float]:
    for _ in range(5):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeat):
        if flush is not None:
            flush.add_(1)
            synchronize(device)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start_ns = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - start_ns) / 1e6)
    return samples


def interleaved_samples_ms(
    providers: dict[str, object],
    device: torch.device,
    repeat: int,
    flush: torch.Tensor | None,
) -> dict[str, list[float]]:
    """Measure providers in a deterministic rotating order.

    Long cold-cache matrices otherwise assign thermal/clock drift to provider
    identity because every implementation is sampled in one contiguous run.
    Rotation gives every provider each order position equally often while
    preserving exact per-sample cache flushing and CUDA-event timing.
    """
    names = tuple(providers)
    for function in providers.values():
        for _ in range(5):
            function()
    synchronize(device)
    samples = {name: [] for name in names}
    for iteration in range(repeat):
        offset = iteration % len(names)
        order = (*names[offset:], *names[:offset])
        for name in order:
            if flush is not None:
                flush.add_(1)
                synchronize(device)
            function = providers[name]
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                function()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
            else:
                start_ns = time.perf_counter_ns()
                function()
                samples[name].append(
                    (time.perf_counter_ns() - start_ns) / 1e6)
    return samples


def median_ms(
    fn, device: torch.device, repeat: int, flush: torch.Tensor | None
) -> float:
    return statistics.median(samples_ms(fn, device, repeat, flush))


def measure_roofs(device: torch.device, quick: bool, repeat: int) -> Roof:
    def copy_bandwidth(tensor_bytes: int) -> float:
        elements = tensor_bytes // 4
        source = torch.randn(elements, device=device)
        destination = torch.empty_like(source)
        copy_ms = median_ms(
            lambda: destination.copy_(source), device, repeat, flush=None)
        traffic = 2 * source.numel() * source.element_size()
        return traffic / (copy_ms * 1e-3) / 1e9

    dram_tensor_bytes = (32 if quick else 256) * 1024 * 1024
    if device.type == "cuda":
        l2_size = torch.cuda.get_device_properties(device).L2_cache_size
        l2_tensor_bytes = max(1 << 20, min(8 << 20, l2_size // 6))
    else:
        l2_tensor_bytes = 1 << 20
    dram_bandwidth = copy_bandwidth(dram_tensor_bytes)
    l2_bandwidth = copy_bandwidth(l2_tensor_bytes)

    size = 2048 if quick else 4096
    left = torch.randn((size, size), device=device)
    right = torch.randn((size, size), device=device)
    output = torch.empty_like(left)
    old_tf32 = None
    if device.type == "cuda":
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        matmul_ms = median_ms(
            lambda: torch.mm(left, right, out=output),
            device,
            max(5, repeat // 2),
            flush=None,
        )
    finally:
        if old_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
    fp32 = (2.0 * size**3) / (matmul_ms * 1e-3) / 1e9
    return Roof(
        dram_bandwidth,
        l2_bandwidth,
        fp32,
        dram_tensor_bytes,
        l2_tensor_bytes,
        size,
    )
