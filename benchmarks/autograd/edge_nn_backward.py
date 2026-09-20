"""Edge-NN training step: fused recompute VJP vs eager Torch autograd.

One forward+backward step of the radius edge-MLP workload from
``benchmarks/neural_networks/radius_edge_mlp.py``. The eager provider
materializes [E, ·] gathers, hidden activations and messages for autograd;
the Tiga provider replays per-edge activations inside the backward
tile kernel and materializes none. Reports median step latency and peak
device memory for both.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import tiga as tg
import torch
from torch import nn

from benchmarks.common.hardware_roofline import samples_ms
from benchmarks.common.output_layout import artifact_path

DIMS = 3
X_WIDTH = 8
HIDDEN = 16
OUT = 8


class EdgeMLP(tg.MessagePassing):
    def __init__(self, module):
        super().__init__()
        self.mlp = tg.nn.trace(module)

    def edge(self, src, dst, edge):
        return self.mlp(edge.displacement, src.x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=131072)
    parser.add_argument(
        "--degree", type=int, default=32,
        help="target average degree inside the cutoff")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.quick:
        args.particles, args.repeat = 16384, 10
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")
    n = args.particles
    generator = torch.Generator().manual_seed(20260824)
    positions = torch.rand((n, DIMS), generator=generator).to(device)
    # Average degree of a uniform unit-cube point set at cutoff r.
    cutoff = (args.degree * 3.0 / (4.0 * torch.pi * n)) ** (1.0 / 3.0)
    x = torch.randn(
        (n, X_WIDTH), generator=torch.Generator().manual_seed(7)).to(device)
    positions.requires_grad_(True)
    x.requires_grad_(True)
    mlp = nn.Sequential(
        nn.Linear(DIMS + X_WIDTH, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, OUT),
    ).to(device)

    graph = tg.Graph.radius(positions, cutoff=cutoff)
    row_ptr, col_idx = graph.resolve_csr()
    edges = col_idx.numel()
    rows = torch.repeat_interleave(
        torch.arange(n, device=device), row_ptr[1:] - row_ptr[:-1])
    cotangent = torch.randn(
        (n, OUT), generator=torch.Generator().manual_seed(11)).to(device)
    program = EdgeMLP(mlp)

    def zero_grads() -> None:
        positions.grad = None
        x.grad = None
        for parameter in mlp.parameters():
            parameter.grad = None

    def tiga_step() -> None:
        zero_grads()
        out = program(graph=graph, src={"x": x}, dst={})
        (out * cotangent).sum().backward()

    def eager_step() -> None:
        zero_grads()
        displacement = positions[col_idx] - positions[rows]
        message = mlp(torch.cat([displacement, x[col_idx]], dim=-1))
        out = torch.zeros(n, OUT, device=device).index_add_(0, rows, message)
        (out * cotangent).sum().backward()

    # Cross-check gradients once before timing.  Field grads reduce over
    # <=degree terms; weight grads reduce over all E edges, so their
    # accumulation-order noise floor is looser.
    tiga_step()
    fused = [x.grad.clone(), positions.grad.clone()]
    fused += [p.grad.clone() for p in mlp.parameters()]
    eager_step()
    eager = [x.grad, positions.grad] + [p.grad for p in mlp.parameters()]
    for name, got, want in zip(
            ("x", "positions", "W0", "b0", "W1", "b1"), fused, eager):
        rtol = 2e-4 if name in ("x", "positions") else 1e-3
        torch.testing.assert_close(got, want, rtol=rtol, atol=1e-5,
                                   msg=lambda e, name=name: f"{name}: {e}")

    providers = {
        "tiga.fused_recompute_vjp": tiga_step,
        "torch.eager_autograd": eager_step,
    }
    results = {}
    for name, step in providers.items():
        raw = samples_ms(step, device, args.repeat, None)
        results[name] = {"milliseconds": statistics.median(raw),
                         "samples_ms": raw}

    # Peak device memory of one training step per provider.
    memory = {}
    for name, step in providers.items():
        zero_grads()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        step()
        torch.cuda.synchronize()
        memory[name] = torch.cuda.max_memory_allocated() - baseline

    eager_bytes = memory["torch.eager_autograd"]
    fused_bytes = memory["tiga.fused_recompute_vjp"]
    eager_ms = results["torch.eager_autograd"]["milliseconds"]
    fused_ms = results["tiga.fused_recompute_vjp"]["milliseconds"]
    print(f"edges={edges}  degree~{edges / n:.1f}")
    for name in providers:
        print(f"{name:32s} {results[name]['milliseconds']:8.3f} ms  "
              f"peak step memory {memory[name] / 2**20:8.1f} MiB")
    print(f"speedup {eager_ms / fused_ms:.2f}x  "
          f"memory {eager_bytes / max(fused_bytes, 1):.2f}x lower")

    if args.json is None:
        args.json = artifact_path(
            "edge_nn_backward", f"cuda_n{n}_degree{args.degree}")
    payload = {
        "operation": "edge_nn_backward",
        "workload": "edge MLP training step (forward + backward)",
        "case": args.json.parent.name,
        "config": {
            "particles": n, "edges": edges, "cutoff": cutoff,
            "mlp": f"{DIMS + X_WIDTH}->{HIDDEN}->{OUT}", "dtype": "float32",
            "timing": "one forward+backward step; compile excluded",
        },
        "results": [
            {"provider": name, **values} for name, values in results.items()],
        "memory": {
            "step_peak_bytes": memory,
            "eager_activation_bytes_estimate": edges * (
                DIMS + X_WIDTH + HIDDEN + OUT) * 4,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
