"""Radius-graph message passing with an edge-local MLP, Tiga vs Warp.

The workload is the PointNet++/continuous-conv message family on a dynamic
radius relation:

    message[e=(j→i)] = MLP([pos[j] − pos[i] ‖ x[j]])     (not hoistable)
    out[i]           = Σ_{e: ‖pos[j] − pos[i]‖ ≤ r} message[e]

The MLP depends on the edge displacement, so no node-wise precompute can
remove the per-edge work; the only way to avoid materializing O(E) message
tensors is a fused traversal kernel.  NVIDIA Warp is the handwritten-fused
reference: its hash-grid neighbor query plus an in-kernel MLP is exactly the
kernel shape Tiga's compiled path does not yet emit for edge-MLP UDFs
(today Tiga falls back to the eager gather → MLP → index_add oracle).

Providers are peers of one workload:

    tiga-eager  Tiga MessagePassing, eager Torch fallback
    torch-eager       the same math written by hand (dispatch-overhead check)
    warp-fused        handwritten fused hash-grid + in-kernel MLP (no O(E)
                      message tensor is ever materialized)
    triton-fused-tile edge-centric tile kernel: each program stages a 128-edge
                      block through registers and evaluates both MLP layers
                      with tl.dot; edges never leave the tile, output is
                      segment-reduced with atomic adds

Topology construction (Warp hash-grid build, Tiga CSR snapshot) is a
one-time cost and is reported separately from the warm per-call latency.

Requires: ``pip install warp-lang``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import tiga as tg
import torch
import triton
import triton.language as tl
from torch import nn

from benchmarks.common.hardware_roofline import interleaved_samples_ms
from benchmarks.common.output_layout import operation_dir

try:
    import warp as wp
except ImportError as _warp_import_error:  # pragma: no cover - env dependent
    wp = None
    _WARP_ERROR = _warp_import_error
else:
    _WARP_ERROR = None

DIMS = 3
F_IN = 8
HID = 16
F_OUT = 8
IN_DIM = DIMS + F_IN

if wp is not None:
    _HID = wp.constant(HID)
    _F_IN = wp.constant(F_IN)

    # --8<-- [start:warp]
    @wp.kernel
    def _radius_edge_mlp_kernel(
        grid: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        x: wp.array(dtype=wp.float32),
        w1: wp.array(dtype=wp.float32),
        b1: wp.array(dtype=wp.float32),
        w2: wp.array(dtype=wp.float32),
        b2: wp.array(dtype=wp.float32),
        cutoff: wp.float32,
        out: wp.array(dtype=wp.float32),
    ):
        i = wp.tid()
        p = points[i]
        # wp.static unrolling is rejected inside dynamic control flow (the
        # neighbor while-loop), so the F_OUT=8 accumulators stay explicit
        # scalars; they are register-resident across the whole traversal.
        # Warp requires float()/int() construction here: literal initializers
        # are compile-time constants and cannot be mutated in a dynamic loop.
        a0 = float(0.0)  # noqa: UP018
        a1 = float(0.0)  # noqa: UP018
        a2 = float(0.0)  # noqa: UP018
        a3 = float(0.0)  # noqa: UP018
        a4 = float(0.0)  # noqa: UP018
        a5 = float(0.0)  # noqa: UP018
        a6 = float(0.0)  # noqa: UP018
        a7 = float(0.0)  # noqa: UP018
        query = wp.hash_grid_query(grid, p, cutoff)
        neighbor = int(0)  # noqa: UP018,RUF046
        r2 = cutoff * cutoff
        while wp.hash_grid_query_next(query, neighbor):
            if neighbor != i:
                d = points[neighbor] - p
                if wp.dot(d, d) <= r2:
                    # Fuse the output layer into the hidden loop: each
                    # activated hidden unit is consumed immediately, so no
                    # per-edge hidden vector ever hits local memory.
                    for h in range(_HID):
                        base = h * 11
                        s = b1[h]
                        s += w1[base] * d[0]
                        s += w1[base + 1] * d[1]
                        s += w1[base + 2] * d[2]
                        for c in range(_F_IN):
                            s += w1[base + 3 + c] * x[neighbor * 8 + c]
                        a = wp.max(s, 0.0)
                        a0 += w2[h] * a
                        a1 += w2[16 + h] * a
                        a2 += w2[32 + h] * a
                        a3 += w2[48 + h] * a
                        a4 += w2[64 + h] * a
                        a5 += w2[80 + h] * a
                        a6 += w2[96 + h] * a
                        a7 += w2[112 + h] * a
                    # Output-layer bias enters once per edge.
                    a0 += b2[0]
                    a1 += b2[1]
                    a2 += b2[2]
                    a3 += b2[3]
                    a4 += b2[4]
                    a5 += b2[5]
                    a6 += b2[6]
                    a7 += b2[7]
        out[i * 8 + 0] = a0
        out[i * 8 + 1] = a1
        out[i * 8 + 2] = a2
        out[i * 8 + 3] = a3
        out[i * 8 + 4] = a4
        out[i * 8 + 5] = a5
        out[i * 8 + 6] = a6
        out[i * 8 + 7] = a7
    # --8<-- [end:warp]


# --8<-- [start:triton]
@triton.jit
def _edge_mlp_tile_kernel(
    dst_per_edge, src_per_edge, positions, x,
    w1, b1, w2, b2, out, num_edges,
    BLOCK_E: tl.constexpr,
):
    """One 128-edge block per program; both MLP layers are tl.dot tiles.

    The contraction dimension is padded 11→16 so tl.dot sees a legal shape;
    padding columns gather zeros and multiply against zero-padded weight rows.
    The output tile is segment-reduced by destination with masked atomic
    adds, so no row pointer and no O(E) message buffer are ever needed.
    """
    pid = tl.program_id(0)
    edges = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    edge_mask = edges < num_edges
    dst = tl.load(dst_per_edge + edges, mask=edge_mask, other=0)
    src = tl.load(src_per_edge + edges, mask=edge_mask, other=0)

    k = tl.arange(0, 16)[None, :]
    rows = edge_mask[:, None]
    # Assemble the [BLOCK_E, 16] input tile from two masked gathers:
    # columns 0-2 are the displacement, columns 3-10 the source feature,
    # columns 11-15 stay zero.
    pos_mask = rows & (k < 3)
    disp = (
        tl.load(positions + src[:, None] * 3 + k, mask=pos_mask, other=0.0)
        - tl.load(positions + dst[:, None] * 3 + k, mask=pos_mask, other=0.0)
    )
    feat = tl.load(
        x + src[:, None] * 8 + (k - 3),
        mask=rows & (k >= 3) & (k < 11), other=0.0)
    inputs = disp + feat

    cols16 = tl.arange(0, 16)
    w1_tile = tl.load(w1 + cols16[:, None] * 16 + cols16[None, :])
    hidden = tl.dot(inputs, w1_tile, input_precision="ieee")
    hidden += tl.load(b1 + cols16)[None, :]
    hidden = tl.maximum(hidden, 0.0)
    w2_tile = tl.load(w2 + cols16[:, None] * 16 + cols16[None, :])
    message = tl.dot(hidden, w2_tile, input_precision="ieee")
    message += tl.load(b2 + cols16)[None, :]

    out_mask = rows & (k < 8)
    tl.atomic_add(
        out + dst[:, None] * 8 + k, message, mask=out_mask, sem="relaxed")
# --8<-- [end:triton]


# --8<-- [start:tiga]
class EdgeMLPMessagePassing(tg.MessagePassing):
    """edge() calls a Torch MLP on [displacement ‖ src feature].

    With ``traced=True`` the module is wrapped in ``tg.nn.trace`` so the
    compiler can prove the edge-NN tile structure and emit the fused kernel;
    with ``traced=False`` the same math stays on the eager fallback.
    """

    def __init__(self, mlp: nn.Module, *, traced: bool = False):
        super().__init__()
        self._traced = traced
        self.mlp = tg.nn.trace(mlp) if traced else mlp

    def edge(self, src, dst, edge):
        del dst
        if self._traced:
            return self.mlp(edge.displacement, src.x)
        return self.mlp(torch.cat([edge.displacement, src.x], dim=-1))
# --8<-- [end:tiga]


def _build_mlp(device: torch.device, seed: int) -> nn.Module:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    mlp = nn.Sequential(
        nn.Linear(IN_DIM, HID), nn.ReLU(), nn.Linear(HID, F_OUT))
    with torch.no_grad():
        for parameter in mlp.parameters():
            parameter.copy_(
                torch.randn(
                    parameter.shape, generator=generator, dtype=torch.float32)
                * (parameter.shape[-1] ** -0.5 if parameter.ndim > 1 else 0.1)
            )
    return mlp.to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=131072)
    parser.add_argument("--degree", type=int, default=32,
                        help="target average degree inside the cutoff")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if wp is None:
        raise SystemExit("install the warp-lang package for this benchmark") \
            from _WARP_ERROR
    if args.quick:
        args.particles, args.repeat = 16384, 10

    device = torch.device("cuda")
    n = args.particles
    generator = torch.Generator(device="cpu").manual_seed(20260820)
    positions = torch.rand((n, DIMS), generator=generator).to(device)
    x = torch.randn((n, F_IN), generator=generator).to(device)
    mlp = _build_mlp(device, seed=20260820)
    # Average degree of a uniform unit-cube point set at cutoff r is
    # N · 4πr³/3; invert for r.
    cutoff = (args.degree * 3.0 / (4.0 * torch.pi * n)) ** (1.0 / 3.0)

    # --- one-time topology builds (reported, never hidden in warm numbers) ---
    started = time.perf_counter_ns()
    graph = tg.Graph.radius(positions, cutoff=cutoff)
    row_ptr, col_idx = graph.resolve_csr()
    destination = graph.destination_index(row_ptr)
    torch.cuda.synchronize()
    gf_topology_ms = (time.perf_counter_ns() - started) / 1e6
    num_edges = col_idx.numel()

    wp.init()
    wp_points = wp.from_torch(positions.contiguous(), dtype=wp.vec3)
    started = time.perf_counter_ns()
    grid_dim = max(8, int(1.0 / cutoff) + 1)
    hash_grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device="cuda")
    hash_grid.build(wp_points, cutoff)
    wp.synchronize()
    warp_topology_ms = (time.perf_counter_ns() - started) / 1e6

    wp_x = wp.from_torch(x.contiguous().reshape(-1))
    wp_out = wp.zeros(n * F_OUT, dtype=wp.float32, device="cuda")
    flat = {
        name: wp.from_torch(parameter.detach().reshape(-1).contiguous())
        for name, parameter in (
            ("w1", mlp[0].weight), ("b1", mlp[0].bias),
            ("w2", mlp[2].weight), ("b2", mlp[2].bias),
        )
    }

    def warp_launch():
        wp.launch(
            _radius_edge_mlp_kernel, dim=n,
            inputs=[
                hash_grid.id, wp_points, wp_x,
                flat["w1"], flat["b1"], flat["w2"], flat["b2"],
                cutoff, wp_out,
            ],
            device="cuda",
        )
        wp.synchronize()

    kernel = EdgeMLPMessagePassing(mlp)

    def gf_eager():
        return kernel(graph=graph, src={"x": x}, dst={})

    compiled_kernel = EdgeMLPMessagePassing(mlp, traced=True)

    def gf_compiled():
        # Inference-mode call: the fused kernel records no autograd tape,
        # so grad-enabled calls deliberately route to the eager oracle.
        with torch.no_grad():
            return compiled_kernel(graph=graph, src={"x": x}, dst={})

    # --8<-- [start:torch]
    def torch_eager():
        src_at_edge = x[col_idx]
        displacement = positions[col_idx] - positions[destination]
        message = mlp(torch.cat([displacement, src_at_edge], dim=-1))
        return torch.zeros(
            (n, F_OUT), dtype=torch.float32, device=device,
        ).index_add_(0, destination, message)
    # --8<-- [end:torch]

    # Triton tile provider: contraction dim padded 11→16 for tl.dot; the
    # destination-per-edge array is one-time topology (like row_ptr).
    dst_i32 = destination.to(torch.int32).contiguous()
    src_i32 = col_idx.to(torch.int32).contiguous()
    w1_pad = torch.zeros((16, 16), dtype=torch.float32, device=device)
    w1_pad[:IN_DIM, :HID] = mlp[0].weight.detach().t()
    w2_pad = torch.zeros((16, 16), dtype=torch.float32, device=device)
    w2_pad[:HID, :F_OUT] = mlp[2].weight.detach().t()
    b1_pad = torch.zeros(16, dtype=torch.float32, device=device)
    b1_pad[:HID] = mlp[0].bias.detach()
    b2_pad = torch.zeros(16, dtype=torch.float32, device=device)
    b2_pad[:F_OUT] = mlp[2].bias.detach()
    tile_out = torch.empty((n, F_OUT), dtype=torch.float32, device=device)
    block_e = 128

    def triton_fused():
        tile_out.zero_()
        _edge_mlp_tile_kernel[(triton.cdiv(num_edges, block_e),)](
            dst_i32, src_i32, positions, x,
            w1_pad, b1_pad, w2_pad, b2_pad, tile_out, num_edges,
            BLOCK_E=block_e, num_warps=4,
        )

    # --- correctness: fused kernels vs eager torch semantics ---
    reference = torch_eager()
    warp_launch()
    warp_output = wp.to_torch(wp_out).reshape(n, F_OUT).clone()
    max_abs_error = (warp_output - reference).abs().max().item()
    if not torch.allclose(warp_output, reference, atol=1e-4, rtol=1e-4):
        raise SystemExit(
            f"warp fused kernel disagrees with eager semantics: "
            f"max |err| = {max_abs_error:.3e}")

    triton_fused()
    tile_error = (tile_out - reference).abs().max().item()
    if not torch.allclose(tile_out, reference, atol=1e-3, rtol=1e-3):
        raise SystemExit(
            f"triton fused tile kernel disagrees with eager semantics: "
            f"max |err| = {tile_error:.3e}")

    gf_output = gf_eager()
    if not torch.allclose(gf_output, reference, atol=1e-4, rtol=1e-4):
        raise SystemExit("tiga eager path disagrees with torch manual")

    compiled_output = gf_compiled()
    compiled_error = (compiled_output - reference).abs().max().item()
    if not torch.allclose(compiled_output, reference, atol=1e-3, rtol=1e-3):
        raise SystemExit(
            f"tiga compiled tile kernel disagrees with eager semantics: "
            f"max |err| = {compiled_error:.3e}")
    oracle_gap = (compiled_output - tile_out).abs().max().item()

    # --- warm latency, interleaved so clock drift is shared fairly ---
    providers = {
        "tiga-eager": gf_eager,
        "torch-eager": torch_eager,
        "warp-fused": warp_launch,
        "triton-fused-tile": triton_fused,
        "tiga-compiled-tile": gf_compiled,
    }
    samples = interleaved_samples_ms(providers, device, args.repeat, flush=None)

    # --- memory: eager materializes O(E) activations; the fused kernel does not
    torch.cuda.reset_peak_memory_stats()
    torch_eager()
    torch.cuda.synchronize()
    eager_peak_bytes = torch.cuda.max_memory_allocated()
    message_bytes = num_edges * (IN_DIM + HID + F_OUT) * 4

    result = {
        "workload": "radius edge MLP",
        "particles": n,
        "target_degree": args.degree,
        "edges": num_edges,
        "measured_degree": num_edges / n,
        "cutoff": cutoff,
        "mlp": {"in": IN_DIM, "hidden": HID, "out": F_OUT},
        "topology_ms": {
            "graphforge_csr_snapshot": gf_topology_ms,
            "warp_hash_grid": warp_topology_ms,
        },
        "results": [
            {
                "provider": name,
                "milliseconds": statistics.median(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "cache": "hot",
                "features": F_OUT,
            }
            for name, values in samples.items()
        ],
        "memory": {
            "eager_peak_bytes": eager_peak_bytes,
            "eager_message_activation_bytes": message_bytes,
            "fused_message_activation_bytes": 0,
        },
        "correctness": {
            "warp_max_abs_error": max_abs_error,
            "triton_tile_max_abs_error": tile_error,
            "compiled_tile_max_abs_error": compiled_error,
            "compiled_vs_handwritten_oracle_gap": oracle_gap,
        },
    }
    print(json.dumps(result, indent=2))

    from benchmarks.common.plotting import plot_latency

    output_dir = args.output_dir or operation_dir("radius_edge_mlp")
    case_dir = Path(output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "results.json").write_text(json.dumps(result, indent=2))
    plot_latency(result, case_dir)
    print(f"wrote {case_dir / 'results.json'}")


if __name__ == "__main__":
    main()
