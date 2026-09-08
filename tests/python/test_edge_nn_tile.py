from __future__ import annotations

import importlib.util
import unittest

import tiga as gf
import torch
from tiga.compiler.capture import Expr
from tiga.compiler.edge_nn_tile import (
    Segment,
    build_edge_nn_spec,
    emit_edge_nn_tile_ttir,
)
from tiga.compiler.nn_capture import (
    ActivationLayer,
    LinearLayer,
    trace,
)
from torch import nn


class EdgeNNCaptureTest(unittest.TestCase):
    def test_sequential_mlp_translates_to_linear_chain(self):
        dag = trace(nn.Sequential(
            nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 8))).dag
        kinds = [type(layer) for layer in dag.layers]
        self.assertEqual(
            kinds, [LinearLayer, ActivationLayer, LinearLayer])
        self.assertEqual(dag.in_features, 11)
        self.assertEqual(dag.out_features, 8)
        first = dag.layers[0]
        self.assertEqual((first.weight_name, first.bias_name), ("0.weight", "0.bias"))

    def test_functional_activations_translate(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                return torch.tanh(torch.nn.functional.gelu(self.proj(x)))

        dag = trace(Net()).dag
        self.assertEqual(
            [getattr(layer, "kind", None) for layer in dag.layers],
            [None, "gelu", "tanh"])

    def test_unsupported_module_is_rejected(self):
        with self.assertRaises(NotImplementedError):
            trace(nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4)))

    def test_branching_graph_is_rejected(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.Linear(4, 4)
                self.right = nn.Linear(4, 4)

            def forward(self, x):
                return self.left(x) + self.right(x)

        with self.assertRaises(NotImplementedError):
            trace(Net())

    def test_traced_module_eager_matches_plain_module(self):
        module = nn.Sequential(nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 8))
        wrapped = trace(module)
        a = torch.randn(7, 3)
        b = torch.randn(7, 8)
        torch.testing.assert_close(
            wrapped(a, b), module(torch.cat([a, b], dim=-1)))

    def test_traced_module_builds_capture_expression(self):
        wrapped = trace(nn.Sequential(nn.Linear(11, 8)))
        message = wrapped(
            Expr("field", ("edge", "displacement")),
            Expr("field", ("src", "x")),
        )
        self.assertEqual(message.op, "nn_subgraph")
        dag, inputs = message.args
        self.assertEqual(len(inputs), 2)
        self.assertEqual(dag.in_features, 11)


class EdgeNNTileSpecTest(unittest.TestCase):
    def _message(self, dag):
        return Expr("nn_subgraph", (dag, (
            Expr("field", ("edge", "displacement")),
            Expr("field", ("src", "x")),
        )))

    def test_spec_from_displacement_and_field(self):
        dag = trace(nn.Sequential(nn.Linear(11, 8))).dag
        spec = build_edge_nn_spec(
            self._message(dag),
            field_widths={("src", "x"): 8},
            position_dim=3,
        )
        self.assertIsNotNone(spec)
        self.assertEqual(spec.segments, (
            Segment("displacement", "displacement", 3, 0),
            Segment("src_field", "x", 8, 3),
        ))
        self.assertEqual((spec.k_pad, spec.out_width, spec.out_pad), (16, 8, 16))
        self.assertTrue(spec.needs_positions)

    def test_width_mismatch_is_rejected(self):
        dag = trace(nn.Sequential(nn.Linear(12, 8))).dag
        self.assertIsNone(build_edge_nn_spec(
            self._message(dag),
            field_widths={("src", "x"): 8},
            position_dim=3,
        ))

    def test_distance_input_is_rejected_in_v1(self):
        dag = trace(nn.Sequential(nn.Linear(1, 4))).dag
        message = Expr("nn_subgraph", (dag, (
            Expr("field", ("edge", "distance")),)))
        self.assertIsNone(build_edge_nn_spec(
            message, field_widths={}, position_dim=3))

    def test_emitter_produces_dot_and_atomic(self):
        dag = trace(nn.Sequential(
            nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 8))).dag
        spec = build_edge_nn_spec(
            self._message(dag),
            field_widths={("src", "x"): 8},
            position_dim=3,
        )
        text = emit_edge_nn_tile_ttir(spec)
        self.assertIn("tt.func public @gf_edge_nn_tile", text)
        self.assertEqual(text.count("tt.dot"), 2)
        self.assertIn("tt.atomic_rmw fadd, relaxed, gpu", text)
        self.assertIn("arith.maxnumf", text)  # relu


class _EdgeMLP(gf.MessagePassing):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = gf.nn.trace(mlp)

    def edge(self, src, dst, edge):
        return self.mlp(edge.displacement, src.x)


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton"),
    "CUDA and Triton are required",
)
class EdgeNNTileExecutionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260820)
        self.device = torch.device("cuda")
        self.positions = torch.rand((256, 3), device=self.device)
        self.x = torch.randn((256, 8), device=self.device)
        self.graph = gf.Graph.radius(self.positions, cutoff=0.25)

    def _run(self, mlp):
        kernel = _EdgeMLP(mlp.to(self.device))
        with torch.no_grad():
            output = kernel(graph=self.graph, src={"x": self.x}, dst={})
        reference = kernel.reference(graph=self.graph, src={"x": self.x}, dst={})
        self.assertTrue(
            torch.allclose(output, reference, atol=1e-3, rtol=1e-3),
            f"max err {(output - reference).abs().max().item():.3e}")
        return kernel, output, reference

    def test_two_layer_mlp_matches_eager(self):
        kernel, _output, _reference = self._run(
            nn.Sequential(nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 8)))
        self.assertEqual(
            kernel.variants[-1].lowering, "gf-python-emit-edge-nn-tile")

    def test_activation_variants_match_eager(self):
        for act in (nn.GELU, nn.Sigmoid, nn.Tanh, nn.SiLU):
            with self.subTest(activation=act.__name__):
                self._run(nn.Sequential(nn.Linear(11, 8), act()))

    def test_bias_free_and_deeper_chains_match_eager(self):
        self._run(nn.Sequential(nn.Linear(11, 8, bias=False)))
        self._run(nn.Sequential(
            nn.Linear(11, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.GELU(),
            nn.Linear(16, 4),
        ))

    def test_grad_enabled_call_falls_back_to_exact_eager(self):
        kernel = _EdgeMLP(nn.Sequential(
            nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 8)).to(self.device))
        x = self.x.clone().requires_grad_(True)
        output = kernel(graph=self.graph, src={"x": x}, dst={})
        self.assertNotEqual(
            kernel.variants[-1].lowering, "gf-python-emit-edge-nn-tile")
        output.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_shape_matrix_matches_eager(self):
        cases = [
            (1, 16, 1),    # scalar field, scalar message
            (13, 16, 8),   # K = 16 exactly: no contraction padding
            (16, 24, 16),  # K = 19 → pad 32; out = 16: no output padding
            (29, 48, 20),  # K = 32 exactly; out = 20 → pad 32
            (8, 16, 32),   # output wider than the hidden layer
        ]
        for f_in, hid, f_out in cases:
            with self.subTest(f_in=f_in, hid=hid, f_out=f_out):
                x = torch.randn((256, f_in), device=self.device)
                kernel = _EdgeMLP(nn.Sequential(
                    nn.Linear(3 + f_in, hid), nn.ReLU(), nn.Linear(hid, f_out),
                ).to(self.device))
                with torch.no_grad():
                    output = kernel(graph=self.graph, src={"x": x}, dst={})
                reference = kernel.reference(
                    graph=self.graph, src={"x": x}, dst={})
                self.assertTrue(
                    torch.allclose(output, reference, atol=1e-3, rtol=1e-3),
                    f"max err {(output - reference).abs().max().item():.3e}")

    def test_architecture_matrix_matches_eager(self):
        class CustomNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.up = nn.Linear(11, 16)
                self.down = nn.Linear(16, 8)

            def forward(self, x):
                return self.down(torch.nn.functional.gelu(self.up(x)))

        architectures = [
            nn.Sequential(nn.Linear(11, 8)),                     # single linear
            nn.Sequential(nn.Linear(11, 16), nn.Linear(16, 8)),  # no activation
            CustomNet(),                                         # custom Module
            nn.Sequential(                                       # deeper chain
                nn.Linear(11, 16), nn.ReLU(),
                nn.Linear(16, 16), nn.Tanh(),
                nn.Linear(16, 8)),
        ]
        for index, mlp in enumerate(architectures):
            with self.subTest(architecture=index):
                self._run(mlp)

    def test_csr_graph_without_positions(self):
        # A plain CSR graph with an MLP over a source feature only:
        # no displacement segment, no positions operand.
        torch.manual_seed(7)
        src_idx = torch.randint(0, 256, (1024,), device=self.device)
        dst_idx = torch.randint(0, 256, (1024,), device=self.device)
        graph = gf.Graph.from_coo(src_idx, dst_idx, num_src=256, num_dst=256)

        class FeatureOnly(gf.MessagePassing):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = gf.nn.trace(mlp)

            def edge(self, src, dst, edge):
                return self.mlp(src.x)

        kernel = FeatureOnly(nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8)).to(self.device))
        with torch.no_grad():
            output = kernel(graph=graph, src={"x": self.x}, dst={})
        reference = kernel.reference(graph=graph, src={"x": self.x}, dst={})
        self.assertTrue(
            torch.allclose(output, reference, atol=1e-3, rtol=1e-3))
        self.assertEqual(
            kernel.variants[-1].lowering, "gf-python-emit-edge-nn-tile")

    def test_plain_module_uses_eager_oracle(self):
        class Plain(gf.MessagePassing):
            def __init__(self, mlp):
                super().__init__()
                self.mlp = mlp

            def edge(self, src, dst, edge):
                return self.mlp(torch.cat([edge.displacement, src.x], dim=-1))

        mlp = nn.Sequential(nn.Linear(11, 16), nn.ReLU(), nn.Linear(16, 8)
                            ).to(self.device)
        traced = _EdgeMLP(mlp)
        plain = Plain(mlp)
        with torch.no_grad():
            expected = traced(graph=self.graph, src={"x": self.x}, dst={})
            actual = plain(graph=self.graph, src={"x": self.x}, dst={})
        self.assertTrue(torch.allclose(expected, actual, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
