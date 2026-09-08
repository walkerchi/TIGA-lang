"""Extended edge-NN op set: every fused op checked against eager autograd.

Each test builds an nn chain using one or more of the extended tile ops
(parameterized activations, scalar-constant arithmetic, feature-wise
LayerNorm), runs the fused tile forward + symbolic-VJP backward, and
compares outputs and all gradients against an eager gather → module →
index_add oracle.  Every test also asserts the fused lowering actually
engaged — a silent eager fallback would make the numeric checks vacuous.
"""

from __future__ import annotations

import unittest

import tiga as gf
import torch
from torch import nn


def _cuda_available() -> bool:
    return torch.cuda.is_available()


class EdgeMLP(gf.MessagePassing):
    def __init__(self, module, **trace_kwargs):
        super().__init__()
        self.mlp = gf.nn.trace(module, **trace_kwargs)

    def edge(self, src, dst, edge):
        return self.mlp(edge.displacement, src.x)


def _problem(nodes=128, dim=3, width=8, cutoff=0.35, seed=7, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    positions = torch.rand(
        nodes, dim, device=device, generator=generator, requires_grad=True)
    x = torch.randn(
        nodes, width, device=device, generator=generator, requires_grad=True)
    graph = gf.Graph.radius(positions, cutoff=cutoff)
    return graph, positions, x


def _eager(graph, positions, x, module, out_width):
    """Reference forward on the same relation with ordinary torch ops."""
    row_ptr, col_idx = graph.resolve_csr()
    rows = torch.repeat_interleave(
        torch.arange(row_ptr.numel() - 1, device=row_ptr.device),
        row_ptr[1:] - row_ptr[:-1])
    displacement = positions[col_idx] - positions[rows]
    message = module(torch.cat([displacement, x[col_idx]], dim=-1))
    return torch.zeros(
        row_ptr.numel() - 1, out_width, device=row_ptr.device
    ).index_add_(0, rows, message)


@unittest.skipUnless(_cuda_available(), "edge nn tile kernels require CUDA")
class EdgeNNOpsTest(unittest.TestCase):
    def _run_pair(self, module, out_width, rtol=5e-4, atol=1e-5, **problem):
        trace_kwargs = problem.pop("trace_kwargs", {})
        for parameter in module.parameters():
            parameter.grad = None  # autograd accumulates; start clean
        graph, positions, x = _problem(**problem)
        program = EdgeMLP(module, **trace_kwargs)
        out = program(graph=graph, src={"x": x}, dst={})
        self.assertIn(
            "gf-python-emit-edge-nn-tile-vjp", program.explain(),
            "fused lowering did not engage; numeric check would be vacuous")
        coeff = torch.randn(
            out.shape, device=out.device,
            generator=torch.Generator(device=out.device).manual_seed(11))
        (out * coeff).sum().backward()
        fused = [x.grad.clone(), positions.grad.clone()]
        fused += [p.grad.clone() for p in module.parameters()]

        for tensor in (x, positions, *module.parameters()):
            tensor.grad = None
        out_ref = _eager(graph, positions, x, module, out_width)
        self.assertTrue(
            torch.allclose(out.detach(), out_ref, rtol=1e-4, atol=1e-5),
            f"forward mismatch: max abs "
            f"{(out.detach() - out_ref).abs().max().item():.3e}")
        (out_ref * coeff).sum().backward()
        eager = [x.grad, positions.grad]
        eager += [p.grad for p in module.parameters()]
        for name, got, want in zip(
                ["x", "positions"]
                + [n for n, _ in module.named_parameters()], fused, eager):
            # Weight grads accumulate over E edges via relaxed atomics; the
            # f32 reordering noise floor is ~1e-4 at this scale (real bugs
            # are orders of magnitude larger).
            tol = (2e-3, 2e-4) if "." in name else (rtol, atol)
            self.assertTrue(
                torch.allclose(got, want, rtol=tol[0], atol=tol[1]),
                f"{name} gradient mismatch: "
                f"max abs {(got - want).abs().max().item():.3e}")
        return program

    def test_parameterized_activations(self):
        for module_factory in (
            lambda: nn.LeakyReLU(0.1),
            lambda: nn.ELU(0.7),
            lambda: nn.Hardtanh(-1.5, 2.0),
            lambda: nn.ReLU6(),
            lambda: nn.Hardsigmoid(),
            lambda: nn.Hardswish(),
            lambda: nn.Mish(),
            lambda: nn.Softplus(beta=1.5, threshold=15.0),
            lambda: nn.SELU(),
        ):
            with self.subTest(module=module_factory()):
                mlp = nn.Sequential(
                    nn.Linear(11, 16), module_factory(),
                    nn.Linear(16, 4)).cuda()
                self._run_pair(mlp, 4)

    def test_identity_is_skipped(self):
        mlp = nn.Sequential(
            nn.Linear(11, 16), nn.Identity(), nn.Tanh(),
            nn.Linear(16, 4)).cuda()
        self._run_pair(mlp, 4)

    def test_scalar_constant_arithmetic(self):
        class Affine(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(11, 16)
                self.fc2 = nn.Linear(16, 4)

            def forward(self, x):
                h = torch.tanh(self.fc1(x))
                h = h * 0.5 + 1.0          # mul_const, add_const
                h = 2.0 - h                # rsub_const
                h = h / 4.0                # div_const
                h = 6.0 / (h + 2.0)        # add_const, rdiv_const
                h = -h                     # neg → mul_const(-1)
                return self.fc2(h) ** 2    # pow_const(2)

        self._run_pair(Affine().cuda(), 4)

    def test_transcendental_ops_on_positive_ranges(self):
        class Transcendental(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(11, 16)
                self.fc2 = nn.Linear(16, 4)

            def forward(self, x):
                h = torch.sigmoid(self.fc1(x)) + 0.5   # (0.5, 1.5)
                h = torch.log(h + 0.5)                 # > 0
                h = torch.sqrt(h)
                h = torch.rsqrt(h + 0.1)
                h = torch.abs(h)
                h = torch.sin(h)
                h = torch.cos(h)
                h = torch.square(h) + 0.5
                return self.fc2(torch.clamp(h, max=4.0))

        self._run_pair(Transcendental().cuda(), 4)

    def test_layernorm_affine_gradients(self):
        mlp = nn.Sequential(
            nn.Linear(11, 16), nn.LayerNorm(16), nn.GELU(),
            nn.Linear(16, 4)).cuda()
        self._run_pair(mlp, 4)

    def test_layernorm_no_affine(self):
        mlp = nn.Sequential(
            nn.Linear(11, 16), nn.LayerNorm(16, elementwise_affine=False),
            nn.Linear(16, 4)).cuda()
        self._run_pair(mlp, 4)

    def test_layernorm_before_first_linear(self):
        mlp = nn.Sequential(
            nn.LayerNorm(11), nn.Linear(11, 16), nn.SiLU(),
            nn.Linear(16, 4)).cuda()
        self._run_pair(mlp, 4)

    def test_block_e_launch_geometry(self):
        for block_e, num_warps in ((64, 2), (128, 4), (256, 8)):
            with self.subTest(block_e=block_e):
                mlp = nn.Sequential(
                    nn.Linear(11, 16), nn.Mish(), nn.Linear(16, 4)).cuda()
                program = self._run_pair(
                    mlp, 4, trace_kwargs={
                        "block_e": block_e, "num_warps": num_warps})
                self.assertIn(f"block_e={block_e}", program.ir("domain"))

    def test_invalid_launch_geometry_rejected(self):
        with self.assertRaises(ValueError):
            gf.nn.trace(nn.Linear(4, 4), block_e=100)
        with self.assertRaises(ValueError):
            gf.nn.trace(nn.Linear(4, 4), block_e=8)
        with self.assertRaises(ValueError):
            gf.nn.trace(nn.Linear(4, 4), num_warps=3)

    def test_unsupported_ops_rejected_at_trace(self):
        with self.assertRaises(NotImplementedError):
            gf.nn.trace(nn.Sequential(nn.Linear(11, 16), nn.BatchNorm1d(16)))
        with self.assertRaises(NotImplementedError):
            gf.nn.trace(nn.Sequential(
                nn.Linear(11, 16), nn.GELU(approximate="tanh")))


if __name__ == "__main__":
    unittest.main()
