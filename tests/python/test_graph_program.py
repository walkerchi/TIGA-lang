from __future__ import annotations

from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import unittest
from unittest import mock

import torch

import graphforge as gf
from graphforge.compiler.toolchain import find_gf_translate


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
            os.environ, {"GRAPHFORGE_TRANSLATE": translator}
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
            os.environ, {"GRAPHFORGE_TRANSLATE": translator}
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


if __name__ == "__main__":
    unittest.main()
