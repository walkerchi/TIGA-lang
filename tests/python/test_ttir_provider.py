from __future__ import annotations

import importlib.util
import os
import unittest
from unittest import mock

import graphforge as gf
import torch
from graphforge.codegen import compile_ttir, prepare_ttir_task_primitive
from graphforge.compiler.toolchain import find_gf_opt, find_gf_translate
from graphforge.interop.torch.compiler_bridge import (
    lower_kernel_to_ttir_plan,
    lower_mlir_stages,
    message_passing_domain_mlir,
)
from graphforge.interop.torch.provider import TorchCudaSubmissionProvider


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


class RadiusDistanceAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


@unittest.skipUnless(
    torch.cuda.is_available() and importlib.util.find_spec("triton"),
    "CUDA and Triton are required",
)
class TTIRProviderTest(unittest.TestCase):
    @staticmethod
    def _tools():
        from pathlib import Path

        return (
            Path(find_gf_opt() or "__missing_gf_opt__"),
            Path(find_gf_translate() or "__missing_gf_translate__"),
        )

    def test_compiler_degree_worklist_primitives_compile_and_execute(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        degrees = torch.tensor(
            [0, 1, 2, 4, 8, 16, 32, 64] * 16,
            device="cuda", dtype=torch.int64,
        )
        row_ptr = torch.cat((
            torch.zeros(1, device="cuda", dtype=torch.int64),
            degrees.cumsum(0),
        ))
        col_idx = torch.arange(
            int(row_ptr[-1]), device="cuda", dtype=torch.int64
        ) % degrees.numel()
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=degrees.numel())
        x = torch.randn(degrees.numel(), device="cuda")
        weight = torch.randn(col_idx.numel(), device="cuda")
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(), graph=graph,
            src={"x": x}, dst={}, edge={"weight": weight}, params={},
            kernel_name="DegreePrimitiveValidation",
        )
        stages = lower_mlir_stages(module, gf_opt=str(gf_opt))
        assert stages.task is not None
        plan = gf.compiler.translate_task_bundle(
            stages.task, gf_translate=str(gf_translate)
        )
        primitives = {
            invocation.task_kind: prepare_ttir_task_primitive(invocation)
            for invocation in plan.invocations
            if invocation.task_kind in {
                "degree-reset", "degree-histogram",
                "degree-prefix", "degree-scatter",
            }
        }
        with self.assertRaisesRegex(ValueError, "at least 129 elements"):
            primitives["degree-histogram"].launch(
                row_ptr=row_ptr[:8],
                row_worklist=torch.empty(137, device="cuda", dtype=torch.int64),
            )
        requirement = next(
            item for item in plan.resources if item.name == "row_worklist"
        )
        worklist = torch.empty(
            requirement.capacity_bytes // 8,
            device="cuda", dtype=torch.int64,
        )
        for kind in (
            "degree-reset", "degree-histogram",
            "degree-prefix", "degree-scatter",
        ):
            arguments = {"row_worklist": worklist}
            if kind in {"degree-histogram", "degree-scatter"}:
                arguments["row_ptr"] = row_ptr
            primitives[kind].launch(**arguments)
        torch.cuda.synchronize()

        buckets = 4
        self.assertEqual(worklist[:buckets + 1].cpu().tolist(),
                         [0, 80, 96, 112, 128])
        self.assertEqual(
            worklist[buckets + 1:2 * buckets + 1].cpu().tolist(),
            [80, 96, 112, 128],
        )
        row_ids = worklist[2 * buckets + 1:]
        torch.testing.assert_close(
            torch.sort(row_ids).values,
            torch.arange(128, device="cuda", dtype=torch.int64),
        )
        for primitive in primitives.values():
            self.assertIn("cubin", primitive.result.artifacts)

        # A frozen relation reuses its versioned materialization. The execute
        # phase is still the compiler-derived bucket DAG, not a Torch-side
        # reconstruction of the schedule.
        execute = plan.bind_phase(
            "execute", lambda invocation: prepare_ttir_task_primitive(invocation)
        )
        output = torch.empty_like(x)
        resources = {
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "src:x": x,
            "edge:weight": weight,
            "row_worklist": worklist,
            "output": output,
        }
        submission = execute.submit(
            TorchCudaSubmissionProvider(), resources
        )
        submission.wait()
        expected = torch.sparse.mm(
            torch.sparse_csr_tensor(
                row_ptr, col_idx, weight,
                size=(degrees.numel(), degrees.numel()),
                check_invariants=False,
            ),
            x[:, None],
        )[:, 0]
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
        self.assertEqual(
            execute.execution_order,
            ("degree-bucket:0", "degree-bucket:1", "degree-bucket:2",
             "degree-bucket:3", "join:0"),
        )

    def test_compiler_ttir_launches_through_runtime_owned_cuda_driver(self):
        nodes, degree = 4096, 16
        row_ptr = torch.arange(
            0, nodes * degree + 1, degree,
            device="cuda", dtype=torch.int64,
        )
        col_idx = torch.randint(
            nodes, (nodes * degree,), device="cuda", dtype=torch.int64
        )
        x = torch.randn(nodes, device="cuda")
        weight = torch.randn(nodes * degree, device="cuda")
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        kernel = WeightedAggregation()
        with mock.patch.dict(os.environ, {"GRAPHFORGE_CUDA_LAUNCHER": "driver"}):
            actual = kernel(
                graph=graph, src={"x": x}, dst={}, edge={"weight": weight}
            )
        destination = torch.arange(nodes, device="cuda").repeat_interleave(degree)
        expected = torch.zeros_like(x).index_add_(
            0, destination, weight * x[col_idx]
        )
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        executor = kernel._torch_executor
        plan = executor._last_executable.runner.__self__
        self.assertIsNotNone(plan.result.driver_launcher)
        self.assertEqual(kernel.last_variant.lowering,
                         "gf-kernel-to-ttir-fixed-csr-weighted-sum")

    def test_tail_split_direct_filter_scans_the_complete_row_domain(self):
        """A compact row count must never shorten a direct-filter grid."""
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        nodes = 512
        generator = torch.Generator(device="cuda").manual_seed(47)
        degrees = torch.cat((
            torch.full((384,), 8, device="cuda", dtype=torch.int64),
            torch.full((102,), 32, device="cuda", dtype=torch.int64),
            torch.full((26,), 64, device="cuda", dtype=torch.int64),
        ))[torch.randperm(nodes, device="cuda", generator=generator)]
        row_ptr = torch.cat((
            torch.zeros(1, device="cuda", dtype=torch.int64),
            degrees.cumsum(0),
        ))
        col_idx = torch.randint(
            0, nodes, (int(row_ptr[-1]),), device="cuda",
            dtype=torch.int64, generator=generator,
        )
        x = torch.randn(nodes, device="cuda", generator=generator)
        weight = torch.randn(
            col_idx.numel(), device="cuda", generator=generator)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(), graph=graph,
            src={"x": x}, dst={}, edge={"weight": weight}, params={},
            kernel_name="TailSplitDomainCoverage",
        )
        stages = lower_mlir_stages(module, gf_opt=str(gf_opt))
        assert stages.task is not None
        plan = gf.compiler.translate_task_bundle(
            stages.task, gf_translate=str(gf_translate)
        )
        buckets = tuple(
            invocation for invocation in plan.invocations
            if invocation.task_kind == "degree-bucket"
        )
        self.assertEqual(
            tuple(bucket.metadata["row_mapping"] for bucket in buckets),
            ("direct-filter", "worklist"),
        )
        # The <=32 bucket contains 486 rows but direct-filter blockM=8 must
        # still launch over all 512 logical rows: ceil(512 / 8) == 64.
        self.assertEqual(buckets[0].metadata["rows_in_bucket"], 486)
        self.assertEqual(buckets[0].metadata["grid_x"], 64)

        compiled = {
            invocation.name: prepare_ttir_task_primitive(invocation)
            for invocation in plan.invocations
            if invocation.task_kind != "join"
        }
        resolver = lambda invocation: compiled[invocation.name]
        requirement = next(
            item for item in plan.resources if item.name == "row_worklist"
        )
        output = torch.full_like(x, float("nan"))
        resources = {
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "src:x": x,
            "edge:weight": weight,
            "row_worklist": torch.empty(
                requirement.capacity_bytes // 8,
                device="cuda", dtype=torch.int64,
            ),
            "output": output,
        }
        provider = TorchCudaSubmissionProvider()
        plan.bind_phase("materialize", resolver).submit(
            provider, resources).wait()
        plan.bind_phase("execute", resolver).submit(
            provider, resources).wait()

        destination = torch.repeat_interleave(
            torch.arange(nodes, device="cuda"), degrees)
        expected = torch.zeros_like(x).index_add_(
            0, destination, weight * x[col_idx])
        self.assertFalse(output.isnan().any().item())
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)

    def test_compiler_chunked_tail_bundle_compiles_and_executes(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        nodes = 512
        generator = torch.Generator(device="cuda").manual_seed(57)
        # Exercise the intended power-law shape: most rows stay on the
        # bounded row tile, only a compact tail is split.
        degrees = torch.full(
            (nodes,), 8, device="cuda", dtype=torch.int64)
        degrees[-26:-2] = 64
        degrees[-2:] = torch.tensor([192, 256], device="cuda")
        degrees = degrees[torch.randperm(
            nodes, device="cuda", generator=generator)]
        row_ptr = torch.cat((
            torch.zeros(1, device="cuda", dtype=torch.int64),
            degrees.cumsum(0),
        ))
        col_idx = torch.randint(
            0, nodes, (int(row_ptr[-1]),), device="cuda",
            dtype=torch.int64, generator=generator,
        )
        x = torch.randn(nodes, device="cuda", generator=generator)
        weight = torch.randn(
            col_idx.numel(), device="cuda", generator=generator)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(), graph=graph,
            src={"x": x}, dst={}, edge={"weight": weight}, params={},
            kernel_name="SplitRowProviderValidation",
        )
        stages = lower_mlir_stages(module, gf_opt=str(gf_opt))
        assert stages.task is not None
        plan = gf.compiler.translate_task_bundle(
            stages.task, gf_translate=str(gf_translate)
        )
        self.assertEqual(
            tuple(invocation.task_kind for invocation in plan.invocations),
            ("degree-histogram", "degree-reset", "degree-prefix",
             "degree-scatter", "degree-bucket", "degree-bucket", "join"),
        )
        buckets = tuple(
            invocation for invocation in plan.invocations
            if invocation.task_kind == "degree-bucket"
        )
        self.assertEqual(
            tuple(bucket.metadata["row_mapping"] for bucket in buckets),
            ("direct-filter", "worklist-chunked"),
        )
        # The exact CDF selects degree 8 as the short-row bound.  A single
        # chunked worklist then handles all 24 degree-64 rows plus the two
        # oversized rows, avoiding a third kernel launch.
        self.assertEqual(buckets[1].metadata["grid_x"], 26)
        compiled = {
            invocation.name: prepare_ttir_task_primitive(invocation)
            for invocation in plan.invocations
            if invocation.task_kind != "join"
        }
        worklist_requirement = next(
            item for item in plan.resources if item.name == "row_worklist"
        )
        self.assertNotIn("row_partials", {
            item.name for item in plan.resources
        })
        output = torch.empty_like(x)
        resources = {
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "src:x": x,
            "edge:weight": weight,
            "row_worklist": torch.empty(
                worklist_requirement.capacity_bytes // 8,
                device="cuda", dtype=torch.int64,
            ),
            "output": output,
        }
        provider = TorchCudaSubmissionProvider()
        resolver = lambda invocation: compiled[invocation.name]
        plan.bind_phase("materialize", resolver).submit(
            provider, resources).wait()
        plan.bind_phase("execute", resolver).submit(
            provider, resources).wait()

        destination = torch.repeat_interleave(
            torch.arange(nodes, device="cuda"), degrees)
        expected = torch.zeros_like(x).index_add_(
            0, destination, weight * x[col_idx])
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
        for executable in compiled.values():
            self.assertIn("cubin", executable.result.artifacts)

    def test_generated_radius_auto_selects_direct_ttir_and_rebinds(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        generator = torch.Generator(device="cuda").manual_seed(37)
        nodes = 256
        positions = torch.rand(nodes, 3, device="cuda", generator=generator)
        x = torch.randn(nodes, device="cuda", generator=generator)
        graph = gf.Graph.radius(positions, cutoff=0.2)
        kernel = RadiusDistanceAggregation()
        with mock.patch.dict(
            "os.environ",
            {
                "GRAPHFORGE_OPT": str(gf_opt),
                "GRAPHFORGE_TRANSLATE": str(gf_translate),
            },
            clear=False,
        ):
            first = kernel(graph=graph, src={"x": x}, dst={"x": x})
            first_expected = kernel.reference(
                graph=graph, src={"x": x}, dst={"x": x})
            positions.add_(0.001)
            second = kernel(graph=graph, src={"x": x}, dst={"x": x})
            second_expected = kernel.reference(
                graph=graph, src={"x": x}, dst={"x": x})

        torch.testing.assert_close(first, first_expected, rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(second, second_expected, rtol=3e-4, atol=3e-4)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-generated-radius-distance-sum",
        )
        self.assertIn('"gf_kernel.generated_launch"', kernel.ir("kernel"))
        self.assertIn("tt.func public @gf_generated_radius_distance_sum",
                      kernel.ir("gf.kernel.ttir"))
        self.assertIn("ptx", kernel.last_variant.artifacts)
        self.assertEqual(len(kernel.variants), 1)

    def test_generated_radius_kernel_ir_compiles_and_launches(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        generator = torch.Generator(device="cuda").manual_seed(31)
        nodes = 128
        positions = torch.rand(nodes, 3, device="cuda", generator=generator)
        x = torch.randn(nodes, device="cuda", generator=generator)
        graph = gf.Graph.radius(positions, cutoff=0.25)
        directory = graph.generated_cell_directory()
        self.assertIsNotNone(directory)
        domain = message_passing_domain_mlir(
            kernel=RadiusDistanceAggregation(),
            graph=graph,
            directory=directory,
            src={"x": x}, dst={}, edge={}, params={},
            kernel_name="DirectGeneratedRadius",
        )
        stages = lower_mlir_stages(domain, gf_opt=gf_opt)
        plan = lower_kernel_to_ttir_plan(
            stages.kernel, gf_translate=gf_translate)
        direct = compile_ttir(
            plan.module, options={"num_warps": plan.num_warps})
        actual = torch.empty_like(x)
        direct.compiled[plan.grid(nodes)](
            directory.cell_ptr,
            directory.particle_order,
            directory.cell_coordinates,
            directory.extents,
            directory.strides,
            directory.neighbor_offsets,
            directory.lattice,
            directory.inverse_lattice,
            positions,
            x,
            actual,
        )
        row_ptr, col_idx = graph.resolve_csr()
        dst = torch.repeat_interleave(
            torch.arange(nodes, device="cuda"), row_ptr[1:] - row_ptr[:-1])
        distance = torch.linalg.vector_norm(
            positions[col_idx] - positions[dst], dim=-1)
        expected = torch.zeros_like(x).index_add_(
            0, dst, distance * x[col_idx])
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        self.assertEqual(plan.entry, "gf_generated_radius_distance_sum")
        self.assertEqual(plan.block_rows, 1)
        self.assertIn("ptx", direct.artifacts)

    def test_periodic_generated_radius_uses_minimum_image_for_box_and_skew(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        generator = torch.Generator(device="cuda").manual_seed(20260813)
        fractional = torch.rand(
            256, 3, device="cuda", generator=generator)
        source = torch.randn(256, device="cuda", generator=generator)
        lattices = (
            torch.ones(3, device="cuda"),
            torch.tensor(
                [[1.0, 0.0, 0.0], [0.35, 1.0, 0.0],
                 [0.0, 0.35, 1.0]], device="cuda"),
        )
        for periodic in lattices:
            lattice = torch.diag(periodic) if periodic.ndim == 1 else periodic
            positions = fractional @ lattice
            graph = gf.Graph.radius(
                positions, cutoff=0.18, periodic=periodic)
            directory = graph.generated_cell_directory()
            self.assertIsNotNone(directory)
            self.assertTrue(directory.periodic)
            kernel = RadiusDistanceAggregation()
            actual = kernel(
                graph=graph, src={"x": source}, dst={"x": source})
            expected = kernel.reference(
                graph=graph, src={"x": source}, dst={"x": source})
            torch.testing.assert_close(
                actual, expected, rtol=3e-4, atol=3e-4)
            self.assertEqual(
                kernel.last_variant.lowering,
                "gf-kernel-to-ttir-generated-radius-distance-sum",
            )
            ttir = kernel.ir("gf.kernel.ttir")
            self.assertIn("%inverse_lattice", ttir)
            self.assertIn("math.floor", ttir)

    def test_generated_radius_derives_fixed_snapshot_geometry_vjp(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        generator = torch.Generator(device="cuda").manual_seed(20260814)
        nodes = 256
        lattice = torch.tensor(
            [[1.0, 0.0, 0.0], [0.25, 1.0, 0.0],
             [0.0, 0.20, 1.0]], device="cuda")
        positions = (
            torch.rand(nodes, 3, device="cuda", generator=generator)
            @ lattice
        ).requires_grad_()
        source = torch.randn(
            nodes, device="cuda", generator=generator, requires_grad=True)
        cotangent = torch.randn(nodes, device="cuda", generator=generator)
        graph = gf.Graph.radius(
            positions, cutoff=0.18, periodic=lattice)
        kernel = RadiusDistanceAggregation()

        actual = kernel(
            graph=graph, src={"x": source}, dst={"x": source})
        actual_gradients = torch.autograd.grad(
            actual, (positions, source), cotangent, retain_graph=True)

        with torch.no_grad():
            row_ptr, col_idx = graph.resolve_csr()
            destination = graph.destination_index(row_ptr)
        displacement = positions[col_idx] - positions[destination]
        fractional = displacement @ torch.linalg.inv(lattice)
        displacement = (fractional - torch.round(fractional)) @ lattice
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        expected = torch.zeros_like(source).index_add(
            0, destination, distance * source[col_idx])
        expected_gradients = torch.autograd.grad(
            expected, (positions, source), cotangent)

        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        for generated, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(
                generated, reference, rtol=3e-4, atol=3e-4)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-generated-radius-distance-sum",
        )
        self.assertIn("fixed-snapshot geometry VJP", kernel.explain())

        # A second differentiable call must not cache distance values carrying
        # the first call's autograd tape.
        repeated = kernel(
            graph=graph, src={"x": source}, dst={"x": source})
        torch.autograd.grad(repeated.sum(), (positions, source))
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-generated-radius-distance-sum",
        )

    def test_generated_radius_adapts_reuse_and_invalidates_on_position_mutation(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        generator = torch.Generator(device="cuda").manual_seed(43)
        nodes = 512
        positions = torch.rand(nodes, 3, device="cuda", generator=generator)
        x = torch.randn(nodes, device="cuda", generator=generator)
        graph = gf.Graph.radius(positions, cutoff=0.20)
        kernel = RadiusDistanceAggregation()
        environment = {
            "GRAPHFORGE_OPT": str(gf_opt),
            "GRAPHFORGE_TRANSLATE": str(gf_translate),
            "GRAPHFORGE_RADIUS_REUSE_THRESHOLD": "2",
        }
        with mock.patch.dict("os.environ", environment, clear=False):
            first = kernel(graph=graph, src={"x": x}, dst={"x": x})
            second = kernel(graph=graph, src={"x": x}, dst={"x": x})
            reused = kernel(graph=graph, src={"x": x}, dst={"x": x})
            expected = kernel.reference(
                graph=graph, src={"x": x}, dst={"x": x})
            torch.testing.assert_close(first, expected, rtol=3e-4, atol=3e-4)
            torch.testing.assert_close(second, expected, rtol=3e-4, atol=3e-4)
            torch.testing.assert_close(reused, expected, rtol=3e-4, atol=3e-4)
            self.assertEqual(
                kernel.last_variant.lowering,
                "adaptive-materialize-radius-snapshot")

            positions.add_(0.01)
            changed = kernel(graph=graph, src={"x": x}, dst={"x": x})
            changed_expected = kernel.reference(
                graph=graph, src={"x": x}, dst={"x": x})
            torch.testing.assert_close(
                changed, changed_expected, rtol=3e-4, atol=3e-4)
            self.assertEqual(
                kernel.last_variant.lowering,
                "gf-kernel-to-ttir-generated-radius-distance-sum")

    def test_scalar_fixed_graph_auto_selects_direct_kernel_ttir(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        nodes, degree = 128, 4
        row_ptr = torch.arange(
            0, nodes * degree + 1, degree,
            device="cuda", dtype=torch.int64)
        col_idx = torch.arange(
            nodes * degree, device="cuda", dtype=torch.int64) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, device="cuda")
        weight = torch.randn(nodes * degree, device="cuda")
        kernel = WeightedAggregation()
        with mock.patch.dict(
            "os.environ",
            {
                "GRAPHFORGE_OPT": str(gf_opt),
                "GRAPHFORGE_TRANSLATE": str(gf_translate),
            },
            clear=False,
        ):
            actual = kernel(
                graph=graph, src={"x": x}, dst={"x": x},
                edge={"weight": weight})
        expected = kernel.reference(
            graph=graph, src={"x": x}, dst={"x": x},
            edge={"weight": weight})

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-fixed-csr-weighted-sum",
        )
        self.assertIn("tt.func public", kernel.ir("gf.kernel.ttir"))
        self.assertIn("ptx", kernel.last_variant.artifacts)

    def test_vector_fixed_graph_auto_selects_feature_tiled_kernel_ttir(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        nodes, degree, features = 257, 16, 16
        row_ptr = torch.arange(
            0, nodes * degree + 1, degree,
            device="cuda", dtype=torch.int64)
        col_idx = torch.randint(
            nodes, (nodes * degree,), device="cuda", dtype=torch.int64)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, features, device="cuda")
        weight = torch.randn(nodes * degree, device="cuda")
        kernel = WeightedAggregation()
        with mock.patch.dict(
            "os.environ",
            {
                "GRAPHFORGE_OPT": str(gf_opt),
                "GRAPHFORGE_TRANSLATE": str(gf_translate),
            },
            clear=False,
        ):
            actual = kernel(
                graph=graph, src={"x": x}, dst={"x": x},
                edge={"weight": weight})
        expected = torch.sparse.mm(
            torch.sparse_csr_tensor(
                row_ptr, col_idx, weight, size=(nodes, nodes)),
            x,
        )

        torch.testing.assert_close(actual, expected, rtol=4e-4, atol=4e-4)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-fixed-csr-weighted-sum",
        )
        self.assertIn("fixed-row-neighbor-feature", kernel.ir("kernel"))
        self.assertIn("tensor<32x16x16xf32>", kernel.ir("gf.kernel.ttir"))
        self.assertIn("ptx", kernel.last_variant.artifacts)

    def test_scalar_bounded_ragged_auto_selects_direct_kernel_ttir(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        nodes = 1024
        degrees = torch.arange(
            nodes, device="cuda", dtype=torch.int64) % 37
        row_ptr = torch.empty(
            nodes + 1, device="cuda", dtype=torch.int64)
        row_ptr[0] = 0
        torch.cumsum(degrees, dim=0, out=row_ptr[1:])
        edges = int(row_ptr[-1])
        col_idx = torch.arange(
            edges, device="cuda", dtype=torch.int64) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, device="cuda")
        weight = torch.randn(edges, device="cuda")
        kernel = WeightedAggregation()
        with mock.patch.dict(
            "os.environ",
            {
                "GRAPHFORGE_OPT": str(gf_opt),
                "GRAPHFORGE_TRANSLATE": str(gf_translate),
            },
            clear=False,
        ):
            actual = kernel(
                graph=graph, src={"x": x}, dst={"x": x},
                edge={"weight": weight})
        expected = kernel.reference(
            graph=graph, src={"x": x}, dst={"x": x},
            edge={"weight": weight})

        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-bounded-ragged-weighted-sum",
        )
        self.assertIn("tensor<32x64xf32>", kernel.ir("gf.kernel.ttir"))
        self.assertIn("ptx", kernel.last_variant.artifacts)

    def test_kernel_ir_translator_compiles_and_launches(self):
        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")

        nodes, degree = 9, 2
        row_ptr = torch.arange(
            0, nodes * degree + 1, degree,
            device="cuda", dtype=torch.int64)
        col_idx = torch.arange(
            nodes * degree, device="cuda", dtype=torch.int64) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, device="cuda")
        weight = torch.randn(nodes * degree, device="cuda")
        domain = message_passing_domain_mlir(
            kernel=WeightedAggregation(),
            graph=graph,
            src={"x": x}, dst={}, edge={"weight": weight}, params={},
            kernel_name="DirectWeightedSum",
        )
        stages = lower_mlir_stages(domain, gf_opt=gf_opt)
        plan = lower_kernel_to_ttir_plan(
            stages.kernel, gf_translate=gf_translate)
        direct = compile_ttir(
            plan.module, options={"num_warps": plan.num_warps})
        actual = torch.empty_like(x)
        direct.compiled[plan.grid(nodes)](
            row_ptr, col_idx, x, weight, actual)
        expected = torch.sparse.mm(
            torch.sparse_csr_tensor(
                row_ptr, col_idx, weight, size=(nodes, nodes)),
            x[:, None],
        )[:, 0]
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected)
        self.assertEqual(plan.block_rows, 16)
        self.assertEqual(
            plan.abi, ("row_ptr", "col_idx", "x", "weight", "out"))
        self.assertIn("ttgir", direct.artifacts)

    def test_serialized_ttir_reenters_vendor_pipeline(self):
        import triton

        gf_opt, gf_translate = self._tools()
        if not gf_opt.is_file() or not gf_translate.is_file():
            self.skipTest("built GraphForge MLIR tools are required")
        nodes, degree = 64, 4
        row_ptr = torch.arange(
            0, nodes * degree + 1, degree,
            device="cuda", dtype=torch.int64)
        col_idx = torch.arange(
            nodes * degree, device="cuda", dtype=torch.int64) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, device="cuda")
        weight = torch.randn(nodes * degree, device="cuda")
        kernel = WeightedAggregation()
        with mock.patch.dict(
            "os.environ",
            {
                "GRAPHFORGE_OPT": str(gf_opt),
                "GRAPHFORGE_TRANSLATE": str(gf_translate),
            },
            clear=False,
        ):
            expected = kernel(
                graph=graph,
                src={"x": x},
                dst={"x": x},
                edge={"weight": weight},
            )

        self.assertEqual(kernel.ir("ttir"), kernel.code("ttir"))
        direct = compile_ttir(kernel.ir("ttir"))
        actual = torch.empty_like(expected)
        direct.compiled[(triton.cdiv(nodes, 16), 1, 1)](
            row_ptr, col_idx, x, weight, actual)
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        self.assertEqual(direct.provider.name, "triton-nvidia")
        self.assertIn("ttgir", direct.artifacts)
        self.assertIn("ptx", direct.artifacts)
        self.assertIn("cubin", direct.artifacts)
        self.assertIsNotNone(direct.worker_compile_ms)
        self.assertIsNotNone(direct.cache_load_ms)

        # A vendor compiler crash is isolated to the persistent worker.  The
        # next identical request starts a new worker and reloads the valid
        # content-addressed disk artifact in this runtime process.
        from graphforge.codegen.compile_worker import _WORKER

        first_pid = _WORKER.pid
        self.assertIsNotNone(first_pid)
        _WORKER.simulate_crash_for_test()
        recovered = compile_ttir(kernel.ir("ttir"))
        self.assertIsNotNone(_WORKER.pid)
        self.assertNotEqual(first_pid, _WORKER.pid)
        self.assertIn("cubin", recovered.artifacts)

    def test_dynamic_knn_rebinds_compiled_fixed_degree_consumer(self):
        nodes, degree = 512, 8
        generator = torch.Generator(device="cuda").manual_seed(91)
        positions = torch.rand(
            nodes, 3, device="cuda", generator=generator)
        source = torch.rand(nodes, device="cuda", generator=generator)
        weight = torch.ones(nodes * degree, device="cuda")
        graph = gf.Graph.knn(positions, degree)
        kernel = WeightedAggregation()

        for step in range(2):
            actual = kernel(
                graph=graph, src={"x": source}, dst={},
                edge={"weight": weight})
            _row_ptr, col_idx = graph.resolve_csr()
            expected = source[col_idx].reshape(nodes, degree).sum(dim=1)
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
            if step == 0:
                positions.add_(
                    1e-3 * torch.randn(
                        positions.shape, device="cuda", generator=generator))

        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-fixed-csr-weighted-sum")
        self.assertEqual(kernel.cache_info["misses"], 1)
        self.assertGreaterEqual(kernel.cache_info["hits"], 1)
        self.assertIn("dynamic_csr_rebind procedural_knn", kernel.ir())
        self.assertIn("tt.store", kernel.code("ttir"))


if __name__ == "__main__":
    unittest.main()
