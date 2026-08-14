from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import graphforge as gf


class AffineAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.conductivity * src.temperature

    def node(self, dst, flux):
        return flux + dst.bias


class Mean(gf.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 0.0, 0.0

    def lift(self, value):
        return value, 1.0

    def combine(self, left, right):
        return left[0] + right[0], left[1] + right[1]

    def finalize(self, state):
        return state[0] / state[1]


class MeanAggregation(gf.MessagePassing):
    reducer = Mean()

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.x)


class OnlineAttention(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.score, src.value)


class UserStableWeightedMean(gf.Reducer):
    name = "deliberately_not_online_softmax"
    associative = True
    commutative = True

    def identity(self):
        return -1.0e30, 0.0, 0.0

    def lift(self, score, value):
        return score, 1.0, value

    def combine(self, left, right):
        maximum = left[0].maximum(right[0])
        left_scale = (left[0] - maximum).exp()
        right_scale = (right[0] - maximum).exp()
        return (
            maximum,
            left_scale * left[1] + right_scale * right[1],
            left_scale * left[2] + right_scale * right[2],
        )

    def finalize(self, state):
        return state[2] / state[1]


class UserStableAttention(gf.MessagePassing):
    reducer = UserStableWeightedMean()

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.score, src.value)


class Product(gf.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 1.0

    def lift(self, value):
        return value

    def combine(self, left, right):
        return left * right


class ProductAggregation(gf.MessagePassing):
    reducer = Product()

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.x)


class EdgeProductAggregation(gf.MessagePassing):
    reducer = Product()

    def edge(self, src, dst, edge):
        del src, dst
        return self.reducer(edge.x)


class AffineComposition(gf.Reducer):
    """Associative but non-commutative composition, right after left."""

    associative = True
    commutative = False

    def identity(self):
        return 1.0, 0.0

    def lift(self, scale, bias):
        return scale, bias

    def combine(self, left, right):
        return right[0] * left[0], right[0] * left[1] + right[1]

    def finalize(self, state):
        return state[1]


class CompositionAggregation(gf.MessagePassing):
    reducer = AffineComposition(deterministic=True)

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.scale, src.bias)


class MessagePassingAutogradTest(unittest.TestCase):
    def setUp(self):
        self.row_ptr = gf.tensor([0, 2, 3, 5], dtype=gf.int64)
        self.col_idx = gf.tensor([0, 2, 1, 0, 1], dtype=gf.int64)
        self.graph = gf.Graph.from_csr(
            self.row_ptr, self.col_idx, num_src=3, validate="full")
        self.temperature = gf.tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.conductivity = gf.tensor(
            [2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
        self.bias = gf.tensor([0.1, 0.2, 0.3], requires_grad=True)

    def _forward(self):
        kernel = AffineAggregation()
        output = kernel(
            graph=self.graph,
            src={"temperature": self.temperature},
            dst={"bias": self.bias},
            edge={"conductivity": self.conductivity},
        )
        return kernel, output

    def test_udf_forward_and_vjp_need_no_user_backward(self):
        kernel, output = self._forward()
        d_temperature, d_conductivity, d_bias = gf.autograd.grad(
            output.sum(),
            (self.temperature, self.conductivity, self.bias),
        )

        self.assertEqual(output.tolist(), [11.100000381469727,
                                           8.199999809265137,
                                           17.299999237060547])
        self.assertEqual(d_temperature.tolist(), [7.0, 10.0, 3.0])
        self.assertEqual(d_conductivity.tolist(), [1.0, 3.0, 2.0, 1.0, 2.0])
        self.assertEqual(d_bias.tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(kernel.last_variant.lowering,
                         "gf-tensor-relation-autograd")

    def test_reverse_graph_preserves_csr_structure(self):
        _, output = self._forward()
        gradient = gf.autograd.grad(output.sum(), self.temperature)
        expression = gradient.expression()

        self.assertIn("csr_expand_rows", expression)
        self.assertIn("segment_sum", expression)
        self.assertNotIn("destination_index", expression)

    def test_forward_and_vjp_lower_to_native_cpu(self):
        old = os.environ.get("GRAPHFORGE_TENSOR_BACKEND")
        os.environ["GRAPHFORGE_TENSOR_BACKEND"] = "native"
        try:
            _, output = self._forward()
            gradient = gf.autograd.grad(output.sum(), self.temperature)
            self.assertEqual(output.tolist(), [11.100000381469727,
                                               8.199999809265137,
                                               17.299999237060547])
            self.assertEqual(gradient.tolist(), [7.0, 10.0, 3.0])
            self.assertEqual(output.execution["backend"], "cpu-llvm-jit")
            self.assertEqual(gradient.execution["backend"], "cpu-llvm-jit")
        except RuntimeError as error:
            if "native GraphForge compiler extension is unavailable" in str(error):
                self.skipTest(str(error))
            raise
        finally:
            if old is None:
                os.environ.pop("GRAPHFORGE_TENSOR_BACKEND", None)
            else:
                os.environ["GRAPHFORGE_TENSOR_BACKEND"] = old

    def test_additive_tuple_udf_reducer_has_automatic_vjp(self):
        x = gf.tensor([1.0, 2.0, 4.0], requires_grad=True)
        kernel = MeanAggregation()

        output = kernel(graph=self.graph, src={"x": x}, dst={})
        dx = gf.autograd.grad(output.sum(), x)

        self.assertEqual(output.tolist(), [2.5, 2.0, 1.5])
        self.assertEqual(dx.tolist(), [1.0, 1.5, 0.5])
        self.assertIn("gf_tensor.div", output.mlir())
        self.assertIn("gf_tensor.div", gf.autograd.grad_mlir(output.sum(), x))
        self.assertIn("proved-componentwise-additive-udf", kernel.explain())

    def test_online_softmax_reducer_has_stable_native_forward_and_vjp(self):
        # Scores around 1000 overflow a naive exp/sum implementation.  The
        # reducer must use its detached row maximum only as a stable shift;
        # score/value gradients still come entirely from Tensor VJP.
        score = gf.tensor([1000.0, 999.0, 998.0], requires_grad=True)
        value = gf.tensor(
            [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]], requires_grad=True)
        kernel = OnlineAttention()
        output = kernel(
            graph=self.graph, src={"score": score, "value": value}, dst={})
        dscore, dvalue = gf.autograd.grad(output.sum(), (score, value))

        expected_output = (
            (1.2384058, 0.1192029),
            (0.0, 2.0),
            (0.7310586, 0.5378829),
        )
        for actual_row, expected_row in zip(output.tolist(), expected_output):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(
            dscore.tolist(), (-0.5115927, 0.1966119, 0.3149807)):
            self.assertAlmostEqual(actual, expected, places=5)
        expected_dvalue = (
            (1.6118556, 1.6118556),
            (1.2689414, 1.2689414),
            (0.1192029, 0.1192029),
        )
        for actual_row, expected_row in zip(dvalue.tolist(), expected_dvalue):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)
        self.assertIn("csr_segment_max_stop_gradient", output.expression())
        self.assertIn("stable-online-softmax", kernel.explain())
        self.assertIn("gf_tensor.exp", gf.autograd.grad_mlir(output.sum(), score))

    def test_user_stable_reducer_gets_structural_forward_and_vjp(self):
        score = gf.tensor([1000.0, 999.0, 998.0], requires_grad=True)
        value = gf.tensor([1.0, -2.0, 3.0], requires_grad=True)
        user_kernel = UserStableAttention()
        builtin_kernel = OnlineAttention()

        user_output = user_kernel(
            graph=self.graph, src={"score": score, "value": value}, dst={})
        builtin_output = builtin_kernel(
            graph=self.graph, src={"score": score, "value": value}, dst={})
        user_grad = gf.autograd.grad(user_output.sum(), (score, value))
        builtin_grad = gf.autograd.grad(builtin_output.sum(), (score, value))

        self.assertEqual(user_output.tolist(), builtin_output.tolist())
        self.assertEqual(user_grad[0].tolist(), builtin_grad[0].tolist())
        self.assertEqual(user_grad[1].tolist(), builtin_grad[1].tolist())
        self.assertIn("proved-stable-weighted-udf", user_kernel.explain())
        self.assertIn("gf_tensor.exp", user_output.mlir())

    def test_user_stable_reducer_preserves_empty_ragged_row_semantics(self):
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 5], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1, 2], dtype=gf.int64),
            num_src=3,
            validate="full",
        )
        output = UserStableAttention()(
            graph=graph,
            src={
                "score": gf.tensor([1000.0, 999.0, 998.0]),
                "value": gf.tensor([1.0, -2.0, 3.0]),
            },
            dst={},
        )
        values = output.tolist()
        self.assertAlmostEqual(values[0], 0.1931757, places=5)
        self.assertTrue(values[1] != values[1])
        self.assertAlmostEqual(values[2], 0.4458758, places=5)

    def test_general_product_reducer_builds_automatic_reduction_tree_vjp(self):
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 5], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1, 2], dtype=gf.int64),
            num_src=3,
            validate="full",
        )
        x = gf.tensor([2.0, 3.0, 5.0], requires_grad=True)
        kernel = ProductAggregation()
        with patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
            output = kernel(graph=graph, src={"x": x}, dst={})
            gradient = gf.autograd.grad(output.sum(), x)
            self.assertEqual(output.tolist(), [6.0, 1.0, 30.0])
            # row0=x0*x1; row2=x0*x1*x2
            self.assertEqual(gradient.tolist(), [18.0, 12.0, 6.0])
        self.assertEqual(output.execution["backend"], "cpu-llvm-jit")
        self.assertEqual(gradient.execution["backend"], "cpu-llvm-jit")
        self.assertIn("proved-product-monoid", kernel.explain())
        self.assertIn("gf_tensor.csr_segment_product", output.mlir(verify=True))
        self.assertIn("gf_tensor.segment_sum", gf.autograd.grad_mlir(
            output.sum(), x))

    def test_general_reducer_tree_vjp_runs_on_native_cuda(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CUDA availability probe requires optional Torch")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        device = "cuda:0"
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 5], dtype=gf.int64, device=device),
            gf.tensor([0, 1, 0, 1, 2], dtype=gf.int64, device=device),
            num_src=3, validate="full",
        )
        x = gf.tensor(
            [2.0, 3.0, 5.0], device=device, requires_grad=True)
        with patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
            output = ProductAggregation()(
                graph=graph, src={"x": x}, dst={})
            gradient = gf.autograd.grad(output.sum(), x)
            self.assertEqual(output.tolist(), [6.0, 1.0, 30.0])
            self.assertEqual(gradient.tolist(), [18.0, 12.0, 6.0])
        self.assertEqual(output.execution["backend"], "cuda-ttir-triton")
        self.assertEqual(gradient.execution["backend"], "cuda-ttir-triton")
        self.assertIn("gf_tensor_segment_sum", gradient.generated_code("ttir"))

    def test_product_reducer_vjp_is_zero_safe_and_fused_on_cuda(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CUDA availability probe requires optional Torch")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        device = "cuda:0"
        graph = gf.Graph.from_csr(
            gf.tensor([0, 3, 5], dtype=gf.int64, device=device),
            gf.tensor([0, 1, 2, 0, 1], dtype=gf.int64, device=device),
            num_src=3, validate="full",
        )
        edge = gf.tensor(
            [2.0, 0.0, 3.0, 4.0, 5.0], device=device,
            requires_grad=True)
        cotangent = gf.tensor([7.0, 11.0], device=device)
        with patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
            output = EdgeProductAggregation()(
                graph=graph, src={}, dst={}, edge={"x": edge})
            gradient = gf.autograd.grad(output, edge, grad_output=cotangent)
            self.assertEqual(output.tolist(), [0.0, 20.0])
            self.assertEqual(gradient.tolist(), [0.0, 42.0, 0.0, 55.0, 44.0])
        self.assertEqual(gradient.execution["backend"], "cuda-ttir-triton")
        self.assertIn(
            "gf_tensor_csr_product_vjp", gradient.generated_code("ttir"))

    def test_general_vector_reducer_has_native_forward_and_vjp(self):
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 5], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1, 2], dtype=gf.int64),
            num_src=3, validate="full",
        )
        x = gf.tensor(
            [[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]],
            requires_grad=True,
        )
        with patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
            output = ProductAggregation()(
                graph=graph, src={"x": x}, dst={})
            gradient = gf.autograd.grad(output.sum(), x)
            self.assertEqual(
                output.tolist(), [[10.0, 21.0], [1.0, 1.0], [110.0, 273.0]])
            self.assertEqual(
                gradient.tolist(), [[60.0, 98.0], [24.0, 42.0], [10.0, 21.0]])
        self.assertEqual(output.execution["backend"], "cpu-llvm-jit")
        self.assertEqual(gradient.execution["backend"], "cpu-llvm-jit")

    def test_general_vector_reducer_vjp_runs_on_native_cuda(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CUDA availability probe requires optional Torch")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        device = "cuda:0"
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 5], dtype=gf.int64, device=device),
            gf.tensor([0, 1, 0, 1, 2], dtype=gf.int64, device=device),
            num_src=3, validate="full",
        )
        x = gf.tensor(
            [[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]],
            device=device, requires_grad=True,
        )
        with patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
            output = ProductAggregation()(
                graph=graph, src={"x": x}, dst={})
            gradient = gf.autograd.grad(output.sum(), x)
            self.assertEqual(
                output.tolist(), [[10.0, 21.0], [1.0, 1.0], [110.0, 273.0]])
            self.assertEqual(
                gradient.tolist(), [[60.0, 98.0], [24.0, 42.0], [10.0, 21.0]])
        self.assertEqual(output.execution["backend"], "cuda-ttir-triton")
        self.assertEqual(gradient.execution["backend"], "cuda-ttir-triton")
        self.assertIn("gf_tensor_segment_sum", gradient.generated_code("ttir"))

    def test_noncommutative_tuple_reducer_preserves_deterministic_order(self):
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 2, 5], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1, 2], dtype=gf.int64),
            num_src=3, validate="full",
        )
        scale = gf.tensor([2.0, 3.0, 5.0], requires_grad=True)
        bias = gf.tensor([1.0, 4.0, 2.0], requires_grad=True)
        with patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
            output = CompositionAggregation()(
                graph=graph,
                src={"scale": scale, "bias": bias}, dst={})
            dscale, dbias = gf.autograd.grad(
                output.sum(), (scale, bias))
            self.assertEqual(output.tolist(), [7.0, 0.0, 37.0])
            self.assertEqual(dscale.tolist(), [0.0, 6.0, 7.0])
            self.assertEqual(dbias.tolist(), [18.0, 6.0, 1.0])

    def test_native_distributed_graph_fails_closed_before_local_execution(self):
        graph = self.graph.halo(gf.DeviceMesh("cpu", 2), depth=1)
        with self.assertRaisesRegex(
            NotImplementedError, "active DistributedRuntime"
        ):
            AffineAggregation()(
                graph=graph,
                src={"temperature": self.temperature},
                dst={"bias": self.bias},
                edge={"conductivity": self.conductivity},
            )


if __name__ == "__main__":
    unittest.main()
