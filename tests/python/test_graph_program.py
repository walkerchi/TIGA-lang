from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import torch

import tiga as gf
from tiga.compiler.toolchain import find_gf_translate


class Weighted(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return src.x * edge.weight


def ring(nodes: int, degree: int):
    row = torch.arange(nodes + 1, dtype=torch.int64) * degree
    col = (
        torch.arange(nodes, dtype=torch.int64)[:, None]
        + torch.arange(1, degree + 1, dtype=torch.int64)
    ).remainder(nodes).flatten()
    return row, col


def _program_of(value):
    """Find the GraphProgram referenced by a Tensor epilogue expression."""
    seen = set()

    def visit(node):
        if id(node) in seen:
            return None
        seen.add(id(node))
        expression = node._expr
        if expression is None:
            return None
        if expression.op == "program_value":
            return expression.attr("program")
        for operand in expression.operands:
            found = visit(operand)
            if found is not None:
                return found
        return None

    return visit(value)


class GraphProgramTest(unittest.TestCase):
    def setUp(self):
        nodes, degree = 32, 4
        row, col = ring(nodes, degree)
        self.graph = gf.Graph.from_csr(row, col, num_src=nodes)
        self.x = torch.randn(nodes)
        self.w0 = torch.randn(nodes * degree)
        self.w1 = torch.randn(nodes * degree)

    def _horizontal(self):
        @gf.program
        def captured():
            return (
                Weighted()(graph=self.graph, src={"x": self.x}, dst={},
                           edge={"weight": self.w0}),
                Weighted()(graph=self.graph, src={"x": self.x}, dst={},
                           edge={"weight": self.w1}),
            )

        return captured()

    def test_native_round_trip_forms_one_multi_result_kernel(self):
        first, second = self._horizontal()
        self.assertIs(first.program, second.program)
        domain = first.program.ir("domain")
        fused = first.program.ir("fused")
        kernel = first.program.ir("kernel")
        self.assertEqual(domain.count('"gf.apply"'), 2)
        self.assertEqual(fused.count('"gf.apply"'), 1)
        self.assertIn("fusion_count = 2", fused)
        self.assertEqual(kernel.count('"gf_kernel.launch"'), 1)
        self.assertIn("block_rows = 32", kernel)
        # The identical source Tensor is one external SSA argument and is
        # projected twice into the product apply.
        self.assertIn("%arg2, %arg3, %arg2, %arg4", fused)

    def test_canonical_hash_is_stable_for_same_program(self):
        first = self._horizontal()[0].semantic_hash
        second = self._horizontal()[0].semantic_hash
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)

    def test_cross_apply_dataflow_and_stale_version_rejection(self):
        program = gf.GraphProgram()
        first = program.apply(
            Weighted(), graph=self.graph, src={"x": self.x}, dst={},
            edge={"weight": self.w0})
        second = program.apply(
            Weighted(), graph=self.graph, src={"x": first}, dst={},
            edge={"weight": self.w1})
        program.outputs(second)
        domain = program.ir("domain")
        self.assertEqual(domain.count('"gf.apply"'), 2)
        self.assertRegex(domain, r'"gf.apply"\([^)]*%[0-9]+, %arg')
        self.assertEqual(program.ir("fused").count('"gf.apply"'), 2)

        stale = replace(first, version=first.version + 1)
        with self.assertRaisesRegex(ValueError, "stale ProgramValue"):
            program.apply(
                Weighted(), graph=self.graph, src={"x": stale}, dst={},
                edge={"weight": self.w1})

    def test_cross_apply_executes_through_runtime_task_dag(self):
        program = gf.GraphProgram()
        first = program.apply(
            Weighted(), graph=self.graph, src={"x": self.x}, dst={},
            edge={"weight": self.w0})
        second = program.apply(
            Weighted(), graph=self.graph, src={"x": first}, dst={},
            edge={"weight": self.w1})
        program.outputs(second)
        actual = program.run()
        row_ptr, col_idx = self.graph.resolve_csr()
        row_ptr = torch.tensor(row_ptr.tolist(), dtype=torch.int64)
        col_idx = torch.tensor(col_idx.tolist(), dtype=torch.int64)
        expected_first = torch.zeros_like(self.x).index_add_(
            0,
            torch.repeat_interleave(
                torch.arange(self.x.shape[0]), row_ptr[1:] - row_ptr[:-1]),
            self.x[col_idx] * self.w0,
        )
        expected = torch.zeros_like(self.x).index_add_(
            0,
            torch.repeat_interleave(
                torch.arange(self.x.shape[0]), row_ptr[1:] - row_ptr[:-1]),
            expected_first[col_idx] * self.w1,
        )
        torch.testing.assert_close(actual, expected)
        self.assertIn("post_fusion=2", program.explain())

    def test_independent_leaves_execute_per_kernel_on_cpu(self):
        # No fused product kernel exists off CUDA: execution falls back to one
        # ordinary kernel per leaf while the fused IR plan stays inspectable.
        @gf.jit
        def two_leaves():
            return (
                Weighted()(graph=self.graph, src={"x": self.x}, dst={},
                           edge={"weight": self.w0}),
                Weighted()(graph=self.graph, src={"x": self.x}, dst={},
                           edge={"weight": self.w1}),
            )

        first, second = two_leaves()
        self.assertIn("post_fusion=1", first.program.explain())
        actual0, actual1 = first.program.run()
        nodes = self.x.shape[0]
        col = torch.tensor(
            self.graph.resolve_csr()[1].tolist(), dtype=torch.int64)
        expected0 = (self.x[col] * self.w0).view(nodes, -1).sum(1)
        expected1 = (self.x[col] * self.w1).view(nodes, -1).sum(1)
        torch.testing.assert_close(actual0, expected0)
        torch.testing.assert_close(actual1, expected1)

    def test_program_leaf_participates_in_tensor_arithmetic(self):
        program = gf.GraphProgram()
        leaf = program.apply(
            Weighted(), graph=self.graph, src={"x": self.x}, dst={},
            edge={"weight": self.w0})
        program.outputs(leaf)

        shifted = leaf + 1.0
        self.assertIsInstance(shifted, gf.Tensor)
        native = gf.tensor([1.0] * self.x.shape[0])
        summed = native + leaf
        self.assertIsInstance(summed, gf.Tensor)

        eager = Weighted()(graph=self.graph, src={"x": self.x}, dst={},
                           edge={"weight": self.w0})
        self.assertEqual(
            shifted.tolist(), (eager + 1.0).tolist())
        self.assertEqual(
            summed.tolist(), (eager + 1.0).tolist())
        self.assertIs(leaf.program, program)

    def test_stacked_program_and_jit_picard_loop_runs(self):
        kernel = Weighted()
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 4], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1], dtype=gf.int64),
            num_src=2,
        )
        weight = gf.tensor([2.0, -1.0, -1.0, 2.0], dtype=gf.float32)

        @gf.program
        @gf.jit
        def stacked(state, iterations):
            for _ in range(iterations):
                state = state + kernel(
                    graph=graph, src={"x": state}, dst={},
                    edge={"weight": weight})
            return state

        result = stacked(gf.tensor([1.0, 0.0], dtype=gf.float32), 2)
        # A = [[2, -1], [-1, 2]]; two Picard steps state += A @ state.
        self.assertEqual(result.tolist(), [10.0, -6.0])
        self.assertIn("gf_control.repeat", result.mlir())

        @gf.jit
        @gf.program
        def stacked_inverse(state, iterations):
            for _ in range(iterations):
                state = state + kernel(
                    graph=graph, src={"x": state}, dst={},
                    edge={"weight": weight})
            return state

        same = stacked_inverse(gf.tensor([1.0, 0.0], dtype=gf.float32), 2)
        self.assertEqual(same.tolist(), [10.0, -6.0])

    def test_jit_auto_captures_and_fuses_sibling_kernel_calls(self):
        graph, x, w0, w1 = self.graph, self.x, self.w0, self.w1

        @gf.jit
        def pair():
            first = Weighted()(graph=graph, src={"x": x}, dst={},
                               edge={"weight": w0})
            second = Weighted()(graph=graph, src={"x": x}, dst={},
                                edge={"weight": w1})
            return first, second

        first, second = pair()
        self.assertIs(first.program, second.program)
        explain = first.program.explain()
        self.assertIn("applies=2", explain)
        self.assertIn("post_fusion=1", explain)

    def test_jit_expression_output_keeps_leaf_fusion(self):
        graph, x, w0, w1 = self.graph, self.x, self.w0, self.w1

        @gf.jit
        def combined():
            first = Weighted()(graph=graph, src={"x": x}, dst={},
                               edge={"weight": w0})
            second = Weighted()(graph=graph, src={"x": x}, dst={},
                                edge={"weight": w1})
            return first + 2.0 * second

        result = combined()
        self.assertIsInstance(result, gf.Tensor)
        program = _program_of(result)
        explain = program.explain()
        self.assertIn("applies=2", explain)
        self.assertIn("post_fusion=1", explain)

    def test_loop_body_kernel_calls_do_not_register_as_program_leaves(self):
        kernel = Weighted()
        graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 4], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1], dtype=gf.int64),
            num_src=2,
        )
        weight = gf.tensor([2.0, -1.0, -1.0, 2.0], dtype=gf.float32)

        @gf.jit
        def smooth_then_lift(value):
            for _ in range(2):
                value = kernel(
                    graph=graph, src={"x": value}, dst={},
                    edge={"weight": weight})
            return kernel(
                graph=graph, src={"x": value}, dst={},
                edge={"weight": weight})

        result = smooth_then_lift(gf.tensor([1.0, 0.0], dtype=gf.float32))
        self.assertIsInstance(result, gf.ProgramValue)
        # Only the straight-line call is a leaf; the two per-iteration
        # applications stay inside the captured loop body.
        self.assertIn("applies=1", result.program.explain())
        # A = [[2, -1], [-1, 2]] applied three times in total.
        self.assertEqual(result.materialize().tolist(), [14.0, -13.0])

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton"),
        "CUDA and Triton are required",
    )
    def test_cross_apply_task_dag_exposes_each_generated_ptx(self):
        translator = find_gf_translate()
        if translator is None:
            self.skipTest("built gf-translate is required")
        nodes, degree = 1024, 8
        row, col = ring(nodes, degree)
        row, col = row.cuda(), col.cuda()
        graph = gf.Graph.from_csr(row, col, num_src=nodes)
        x = torch.randn(nodes, device="cuda")
        w0 = torch.randn(nodes * degree, device="cuda")
        w1 = torch.randn(nodes * degree, device="cuda")
        program = gf.GraphProgram()
        first = program.apply(
            Weighted(), graph=graph, src={"x": x}, dst={},
            edge={"weight": w0})
        second = program.apply(
            Weighted(), graph=graph, src={"x": first}, dst={},
            edge={"weight": w1})
        program.outputs(second)
        with mock.patch.dict(
            os.environ, {"TIGA_TRANSLATE": translator}
        ):
            actual = program.run()
            expected_first = (x[col] * w0).view(nodes, degree).sum(1)
            expected = (expected_first[col] * w1).view(nodes, degree).sum(1)
            torch.testing.assert_close(actual, expected)
            artifacts = program.code("ptx")
        self.assertEqual(set(artifacts), {"apply_0.ptx", "apply_1.ptx"})
        for ptx in artifacts.values():
            self.assertIn("Generated by LLVM NVPTX", ptx)

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton"),
        "CUDA and Triton are required",
    )
    def test_observation_auto_jits_fused_program_and_exposes_ptx(self):
        translator = find_gf_translate()
        if translator is None:
            self.skipTest("built gf-translate is required")
        nodes, degree = 1024, 8
        row, col = ring(nodes, degree)
        row, col = row.cuda(), col.cuda()
        graph = gf.Graph.from_csr(row, col, num_src=nodes)
        x = torch.randn(nodes, device="cuda")
        w0 = torch.randn(nodes * degree, device="cuda")
        w1 = torch.randn(nodes * degree, device="cuda")

        with mock.patch.dict(
            os.environ, {"TIGA_TRANSLATE": translator}
        ):
            @gf.program
            def captured():
                return (
                    Weighted()(graph=graph, src={"x": x}, dst={},
                               edge={"weight": w0}),
                    Weighted()(graph=graph, src={"x": x}, dst={},
                               edge={"weight": w1}),
                )

            first, second = captured()
            actual0, actual1 = first.program.run()
            expected0 = (x[col] * w0).view(nodes, degree).sum(1)
            expected1 = (x[col] * w1).view(nodes, degree).sum(1)
            torch.testing.assert_close(actual0, expected0)
            torch.testing.assert_close(actual1, expected1)
            self.assertIn("Generated by LLVM NVPTX", first.program.code("ptx"))
            self.assertIn("post_fusion=1", second.program.explain())
    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton"),
        "CUDA and Triton are required",
    )
    def test_jit_auto_fusion_and_epilogue_match_eager_kernels(self):
        translator = find_gf_translate()
        if translator is None:
            self.skipTest("built gf-translate is required")
        nodes, degree = 1024, 8
        row, col = ring(nodes, degree)
        row, col = row.cuda(), col.cuda()
        graph = gf.Graph.from_csr(row, col, num_src=nodes)
        x = torch.randn(nodes, device="cuda")
        w0 = torch.randn(nodes * degree, device="cuda")
        w1 = torch.randn(nodes * degree, device="cuda")

        with mock.patch.dict(
            os.environ, {"TIGA_TRANSLATE": translator}
        ):
            @gf.jit
            def combined():
                first = Weighted()(graph=graph, src={"x": x}, dst={},
                                   edge={"weight": w0})
                second = Weighted()(graph=graph, src={"x": x}, dst={},
                                    edge={"weight": w1})
                return first + 2.0 * second

            result = combined()
            program = _program_of(result)
            self.assertIn("applies=2", program.explain())
            self.assertIn("post_fusion=1", program.explain())
            actual = result.realize().to_torch(copy=True)
            eager0 = Weighted()(graph=graph, src={"x": x}, dst={},
                                edge={"weight": w0})
            eager1 = Weighted()(graph=graph, src={"x": x}, dst={},
                                edge={"weight": w1})
            torch.testing.assert_close(actual, eager0 + 2.0 * eager1)


class GatherSource(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


class ProgramAutogradTest(unittest.TestCase):
    """Reverse mode composes through program leaves: per-leaf inline VJP."""

    def setUp(self):
        self.graph = gf.Graph.from_csr(
            gf.tensor([0, 2, 4], dtype=gf.int64),
            gf.tensor([0, 1, 0, 1], dtype=gf.int64),
            num_src=2,
        )
        self.w0 = gf.tensor([2.0, -1.0, -1.0, 2.0], dtype=gf.float32)
        self.w1 = gf.tensor([1.0, 1.5, 0.5, -0.5], dtype=gf.float32)
        self.x = gf.tensor([1.0, 2.0], dtype=gf.float32, requires_grad=True)

    def _eager(self, value):
        first = Weighted()(graph=self.graph, src={"x": value}, dst={},
                           edge={"weight": self.w0})
        second = Weighted()(graph=self.graph, src={"x": value}, dst={},
                            edge={"weight": self.w1})
        return first, second

    def test_shared_input_leaves_accumulate_and_match_eager(self):
        graph, w0, w1 = self.graph, self.w0, self.w1

        @gf.jit
        def pair(x):
            first = Weighted()(graph=graph, src={"x": x}, dst={},
                               edge={"weight": w0})
            second = Weighted()(graph=graph, src={"x": x}, dst={},
                                edge={"weight": w1})
            return first, second

        first, second = pair(self.x)
        self.assertTrue(first._as_tensor().requires_grad)
        loss = (first * 1.0).sum() + 2.0 * (second * 1.0).sum()
        gradient = gf.autograd.grad(loss, self.x)

        eager_first, eager_second = self._eager(self.x)
        eager_gradient = gf.autograd.grad(
            eager_first.sum() + 2.0 * eager_second.sum(), self.x)
        self.assertEqual(gradient.tolist(), eager_gradient.tolist())

        def scalar_loss(values):
            probe = gf.tensor(values, dtype=gf.float32)
            eager_first, eager_second = self._eager(probe)
            return (eager_first.sum() + 2.0 * eager_second.sum()).tolist()

        eps = 1.0e-3
        for index in range(2):
            plus, minus = [1.0, 2.0], [1.0, 2.0]
            plus[index] += eps
            minus[index] -= eps
            difference = (scalar_loss(plus) - scalar_loss(minus)) / (2 * eps)
            self.assertAlmostEqual(gradient.tolist()[index], difference, places=3)

    def test_epilogue_expression_gradient_matches_eager(self):
        graph, w0, w1 = self.graph, self.w0, self.w1

        @gf.jit
        def combined(x):
            first = Weighted()(graph=graph, src={"x": x}, dst={},
                               edge={"weight": w0})
            second = Weighted()(graph=graph, src={"x": x}, dst={},
                                edge={"weight": w1})
            return (first + second) * 2.0

        loss = combined(self.x).sum()
        gradient = gf.autograd.grad(loss, self.x)
        eager_first, eager_second = self._eager(self.x)
        eager_gradient = gf.autograd.grad(
            ((eager_first + eager_second) * 2.0).sum(), self.x)
        self.assertEqual(gradient.tolist(), eager_gradient.tolist())

    def test_dependent_leaves_compose_their_vjps(self):
        program = gf.GraphProgram()
        first = program.apply(
            Weighted(), graph=self.graph, src={"x": self.x}, dst={},
            edge={"weight": self.w0})
        second = program.apply(
            Weighted(), graph=self.graph, src={"x": first}, dst={},
            edge={"weight": self.w1})
        program.outputs(second)

        gradient = gf.autograd.grad((second * 2.0).sum(), self.x)
        eager_first, _ = self._eager(self.x)
        eager_second = Weighted()(graph=self.graph, src={"x": eager_first},
                                  dst={}, edge={"weight": self.w1})
        eager_gradient = gf.autograd.grad((eager_second * 2.0).sum(), self.x)
        self.assertEqual(gradient.tolist(), eager_gradient.tolist())

    def test_torch_field_leaf_fails_closed(self):
        graph = gf.Graph.from_csr(
            torch.tensor([0, 2, 4]), torch.tensor([0, 1, 0, 1]), num_src=2)
        program = gf.GraphProgram()
        leaf = program.apply(
            Weighted(), graph=graph,
            src={"x": torch.tensor([1.0, 2.0])}, dst={},
            edge={"weight": torch.ones(4)})
        program.outputs(leaf)
        with self.assertRaisesRegex(NotImplementedError, "native"):
            gf.autograd.grad((leaf * 1.0).sum(), self.x)

    def test_unsupported_relation_leaf_fails_closed(self):
        x = gf.tensor([1.0, 2.0, 3.0], dtype=gf.float32, requires_grad=True)
        program = gf.GraphProgram()
        leaf = program.apply(
            GatherSource(), graph=gf.Graph.dense(3),
            src={"x": x}, dst={}, edge={})
        program.outputs(leaf)
        with self.assertRaisesRegex(NotImplementedError, "CSR"):
            gf.autograd.grad((leaf * 1.0).sum(), x)


if __name__ == "__main__":
    unittest.main()
