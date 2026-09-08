"""GAT-style edge attention: fused row-centric kernels vs eager Torch.

One forward+backward training step of graph attention with an nn-produced
edge score on a radius relation. The eager provider materializes [E] scores,
[E] weights and [E, F] weighted messages for autograd; the Tiga
provider keeps the online-softmax state (m, l) per row — O(N) — and replays
the score chain inside the backward tile kernel. Reports median step latency
and peak device memory for both.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import tiga as gf
import torch
from torch import nn

from benchmarks.common.hardware_roofline import samples_ms
from benchmarks.common.output_layout import artifact_path

DIMS = 3
X_WIDTH = 8
HIDDEN = 16


class GAT(gf.MessagePassing):
    reducer = gf.online_softmax()

    def __init__(self, attn):
        super().__init__()
        self.attn = gf.nn.trace(attn)

    def edge(self, src, dst, edge):
        score = self.attn(edge.displacement, src.x, dst.x)
        return self.reducer(score, src.x)


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
    generator = torch.Generator().manual_seed(20260825)
    positions = torch.rand((n, DIMS), generator=generator).to(device)
    # Average degree of a uniform unit-cube point set at cutoff r.
    cutoff = (args.degree * 3.0 / (4.0 * torch.pi * n)) ** (1.0 / 3.0)
    x = torch.randn(
        (n, X_WIDTH), generator=torch.Generator().manual_seed(7)).to(device)
    positions.requires_grad_(True)
    x.requires_grad_(True)
    attn = nn.Sequential(
        nn.Linear(DIMS + 2 * X_WIDTH, HIDDEN), nn.LeakyReLU(0.2),
        nn.Linear(HIDDEN, 1),
    ).to(device)

    graph = gf.Graph.radius(positions, cutoff=cutoff)
    row_ptr, col_idx = graph.resolve_csr()
    edges = col_idx.numel()
    rows = torch.repeat_interleave(
        torch.arange(n, device=device), row_ptr[1:] - row_ptr[:-1])
    cotangent = torch.randn(
        (n, X_WIDTH), generator=torch.Generator().manual_seed(11)).to(device)
    program = GAT(attn)

    def zero_grads() -> None:
        positions.grad = None
        x.grad = None
        for parameter in attn.parameters():
            parameter.grad = None

    def graphforge_step() -> None:
        zero_grads()
        out = program(graph=graph, src={"x": x}, dst={"x": x})
        (out * cotangent).sum().backward()

    def eager_step() -> None:
        zero_grads()
        displacement = positions[col_idx] - positions[rows]
        score = attn(torch.cat(
            [displacement, x[col_idx], x[rows]], dim=-1)).squeeze(-1)
        with torch.no_grad():
            maximum = torch.full(
                (n,), float("-inf"), device=device
            ).scatter_reduce_(0, rows, score.detach(), reduce="amax",
                              include_self=True)
        weight = torch.exp(score - maximum[rows])
        denominator = torch.zeros(n, device=device).index_add(
            0, rows, weight)
        numerator = torch.zeros(n, X_WIDTH, device=device).index_add(
            0, rows, weight.unsqueeze(-1) * x[col_idx])
        out = torch.where(
            denominator.unsqueeze(-1) > 0,
            numerator / denominator.unsqueeze(-1).clamp(min=1e-30),
            torch.zeros_like(numerator))
        (out * cotangent).sum().backward()

    # Cross-check gradients once before timing.  Field grads reduce over
    # <=degree terms; weight grads reduce over all E edges, so their
    # accumulation-order noise floor is looser.
    graphforge_step()
    fused = [x.grad.clone(), positions.grad.clone()]
    fused += [p.grad.clone() for p in attn.parameters()]
    eager_step()
    eager = [x.grad, positions.grad] + [p.grad for p in attn.parameters()]
    for name, got, want in zip(
            ("x", "positions", "W0", "b0", "W1", "b1"), fused, eager):
        torch.testing.assert_close(got, want, rtol=2e-3, atol=2e-4,
                                   msg=lambda e, name=name: f"{name}: {e}")

    providers = {
        "tiga.fused_online_softmax": graphforge_step,
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
    fused_bytes = memory["tiga.fused_online_softmax"]
    eager_ms = results["torch.eager_autograd"]["milliseconds"]
    fused_ms = results["tiga.fused_online_softmax"]["milliseconds"]
    print(f"edges={edges}  degree~{edges / n:.1f}")
    for name in providers:
        print(f"{name:32s} {results[name]['milliseconds']:8.3f} ms  "
              f"peak step memory {memory[name] / 2**20:8.1f} MiB")
    print(f"speedup {eager_ms / fused_ms:.2f}x  "
          f"memory {eager_bytes / max(fused_bytes, 1):.2f}x lower")

    if args.json is None:
        args.json = artifact_path(
            "gat_attention", f"cuda_n{n}_degree{args.degree}")
    payload = {
        "operation": "gat_attention",
        "workload": "GAT edge-attention training step (forward + backward)",
        "case": args.json.parent.name,
        "config": {
            "particles": n, "edges": edges, "cutoff": cutoff,
            "attn": f"{DIMS + 2 * X_WIDTH}->{HIDDEN}->1",
            "value": X_WIDTH, "dtype": "float32",
            "timing": "one forward+backward step; compile excluded",
        },
        "results": [
            {"provider": name, **values} for name, values in results.items()],
        "memory": {
            "step_peak_bytes": memory,
            "eager_activation_bytes_estimate": edges * (
                DIMS + 2 * X_WIDTH + HIDDEN + 2 + X_WIDTH) * 4,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
