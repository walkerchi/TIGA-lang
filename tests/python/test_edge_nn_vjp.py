"""Fused edge-NN tile backward (phase 2): gradients, shapes, fallbacks.

The training path runs the fused tile forward under an autograd bridge whose
backward is the symbolic-VJP recompute kernel; every check compares against
an eager gather → MLP → index_add oracle, the semantic ground truth.
"""

from __future__ import annotations

import unittest

import tiga as tg
import torch
from torch import nn


def _cuda_available() -> bool:
    return torch.cuda.is_available()


class EdgeMLP(tg.MessagePassing):
    def __init__(self, module):
        super().__init__()
        self.mlp = tg.nn.trace(module)

    def edge(self, src, dst, edge):
        return self.mlp(edge.displacement, src.x)


def _problem(nodes=128, dim=3, width=8, cutoff=0.35, seed=7, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(seed)
    positions = torch.rand(
        nodes, dim, device=device, generator=generator, requires_grad=True)
    x = torch.randn(
        nodes, width, device=device, generator=generator, requires_grad=True)
    graph = tg.Graph.radius(positions, cutoff=cutoff)
    return graph, positions, x


def _eager(graph, positions, x, mlp, out_width):
    """Reference forward on the same relation with ordinary torch ops."""
    row_ptr, col_idx = graph.resolve_csr()
    rows = torch.repeat_interleave(
        torch.arange(row_ptr.numel() - 1, device=row_ptr.device),
        row_ptr[1:] - row_ptr[:-1])
    displacement = positions[col_idx] - positions[rows]
    message = mlp(torch.cat([displacement, x[col_idx]], dim=-1))
    return torch.zeros(
        row_ptr.numel() - 1, out_width, device=row_ptr.device
    ).index_add_(0, rows, message)


def _assert_grads_close(test_case, fused, eager, rtol=2e-4, atol=1e-5):
    test_case.assertEqual(len(fused), len(eager))
    names = ["x", "positions"] + [f"parameter[{i}]" for i in range(len(fused) - 2)]
    for name, got, want in zip(names, fused, eager):
        if want is None:
            test_case.assertIsNone(got, name)
            continue
        test_case.assertIsNotNone(got, name)
        test_case.assertTrue(
            torch.allclose(got, want, rtol=rtol, atol=atol),
            f"{name} gradient mismatch: "
            f"max abs {(got - want).abs().max().item():.3e}")


@unittest.skipUnless(_cuda_available(), "edge nn tile kernels require CUDA")
class EdgeNNVjpTest(unittest.TestCase):
    def _run_pair(self, mlp, out_width, **problem):
        for parameter in mlp.parameters():
            parameter.grad = None  # autograd accumulates; start clean
        graph, positions, x = _problem(**problem)
        program = EdgeMLP(mlp)
        out = program(graph=graph, src={"x": x}, dst={})
        coeff = torch.randn(
            out.shape, device=out.device,
            generator=torch.Generator(device=out.device).manual_seed(11))
        (out * coeff).sum().backward()
        fused = [x.grad.clone(), positions.grad.clone()]
        fused += [p.grad.clone() for p in mlp.parameters()]

        for tensor in (x, positions, *mlp.parameters()):
            tensor.grad = None
        out_ref = _eager(graph, positions, x, mlp, out_width)
        self.assertTrue(torch.allclose(
            out.detach(), out_ref, rtol=1e-4, atol=1e-5))
        (out_ref * coeff).sum().backward()
        eager = [x.grad, positions.grad]
        eager += [p.grad for p in mlp.parameters()]
        _assert_grads_close(self, fused, eager)
        return program

    def test_gelu_two_layer_gradients(self):
        mlp = nn.Sequential(nn.Linear(11, 16), nn.GELU(), nn.Linear(16, 4)).cuda()
        program = self._run_pair(mlp, 4)
        self.assertIn("gf-python-emit-edge-nn-tile-vjp", program.explain())

    def test_relu_no_bias_gradients(self):
        mlp = nn.Sequential(
            nn.Linear(11, 16, bias=False), nn.ReLU(),
            nn.Linear(16, 4, bias=False)).cuda()
        self._run_pair(mlp, 4)

    def test_tanh_three_layer_gradients(self):
        mlp = nn.Sequential(
            nn.Linear(11, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.SiLU(),
            nn.Linear(16, 4)).cuda()
        self._run_pair(mlp, 4, nodes=96, cutoff=0.4)

    def test_sigmoid_scalar_field(self):
        mlp = nn.Sequential(nn.Linear(4, 16), nn.Sigmoid(), nn.Linear(16, 1)).cuda()
        self._run_pair(mlp, 1, width=1)

    def test_second_call_after_weight_update(self):
        mlp = nn.Sequential(nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 4)).cuda()
        self._run_pair(mlp, 4)
        with torch.no_grad():
            for parameter in mlp.parameters():
                parameter.add_(0.01)
        self._run_pair(mlp, 4)

    def test_unsupported_udf_still_trains_via_eager(self):
        # edge.distance is outside the tile proof envelope: the call falls
        # back to the eager oracle and autograd still flows.
        class DistanceNet(tg.MessagePassing):
            def __init__(self, module):
                super().__init__()
                self.mlp = tg.nn.trace(module)

            def edge(self, src, dst, edge):
                return self.mlp(edge.distance.unsqueeze(-1), src.x)

        mlp = nn.Sequential(nn.Linear(9, 16), nn.ReLU(), nn.Linear(16, 4)).cuda()
        graph, _positions, x = _problem()
        out = DistanceNet(mlp)(graph=graph, src={"x": x}, dst={})
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class EdgeNNCPUGradientTest(unittest.TestCase):
    def test_inputs_edge_fields_and_parameters_match_torch_and_finite_difference(self):
        """Fixed CSR avoids differentiating a discrete neighbor selection."""
        torch.manual_seed(123)
        module = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()

        class Program(tg.MessagePassing):
            reducer = tg.sum()

            def __init__(self):
                super().__init__()
                self.net = tg.nn.trace(module)

            def edge(self, src, dst, edge):
                return self.net(src.x, edge.attr)

        rows = torch.tensor([0, 2, 3, 3], dtype=torch.int64)
        cols = torch.tensor([0, 1, 1], dtype=torch.int64)
        graph = tg.Graph.from_csr(rows, cols, num_src=2)
        x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        attr = torch.randn(3, 1, dtype=torch.float64, requires_grad=True)
        program = Program()

        def forward(x, attr):
            return program(graph=graph, src={"x": x}, dst={}, edge={"attr": attr})

        output = forward(x, attr)
        destination = torch.tensor([0, 0, 1])
        reference = torch.zeros(3, 2, dtype=torch.float64).index_add(
            0, destination, module(torch.cat([x[cols], attr], dim=-1)))
        torch.testing.assert_close(output, reference, rtol=1e-10, atol=1e-10)
        cotangent = torch.randn_like(output)
        inputs = (x, attr, *module.parameters())
        actual = torch.autograd.grad((output * cotangent).sum(), inputs)
        expected = torch.autograd.grad((reference * cotangent).sum(), inputs)
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10)
        self.assertTrue(torch.autograd.gradcheck(forward, (x, attr)))
        # Check a weight derivative independently of either autograd graph.
        parameter = next(module.parameters())
        original = parameter[0, 0].item()
        epsilon = 1e-6
        try:
            with torch.no_grad():
                parameter[0, 0] = original + epsilon
                plus = (forward(x, attr) * cotangent).sum().item()
                parameter[0, 0] = original - epsilon
                minus = (forward(x, attr) * cotangent).sum().item()
        finally:
            with torch.no_grad():
                parameter[0, 0] = original
        self.assertAlmostEqual(actual[2][0, 0].item(), (plus - minus) / (2 * epsilon), places=7)


if __name__ == "__main__":
    unittest.main()
