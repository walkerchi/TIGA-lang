from __future__ import annotations

import os
import unittest
from unittest import mock

import torch

import graphforge as gf
from graphforge.compiler.toolchain import find_gf_opt, find_gf_translate
from graphforge.interop.torch.compiler_bridge import (
    lower_kernel_to_ttir,
    lower_mlir_stages,
    message_passing_domain_mlir,
)


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.weight * src.x


class RadiusDistanceAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst
        return edge.distance * src.x


class DenseSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


class MeanReducer(gf.Reducer):
    name = "mean"
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


class DenseMean(gf.MessagePassing):
    reducer = MeanReducer()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


class ScalarOnlineReducer(gf.Reducer):
    name = "scalar_online_weighted_mean"
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


class ScalarOnlineAggregation(gf.MessagePassing):
    reducer = ScalarOnlineReducer()

    def edge(self, src, dst, edge):
        del dst, edge
        return self.reducer(src.score, src.value)


class DiffusionUpdate(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.conductivity * (src.u - dst.u)

    def node(self, dst, flux, dt):
        return dst.u + dt * flux


class DenseDiffusionUpdate(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge, scale):
        del edge
        return (src.u - dst.u) * scale

    def node(self, dst, flux, bias):
        return dst.u + flux + bias


class StructuredDense(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        del edge
        return self.reducer(
            (src.right * dst.left).sum(dim=-1) * scale,
            src.payload,
        )


def _gf_opt() -> str | None:
    return find_gf_opt()


def _gf_translate() -> str | None:
    return find_gf_translate()


@unittest.skipUnless(_gf_opt(), "a built gf-opt is required")
class MLIRBridgeTest(unittest.TestCase):
    def test_native_capture_lowers_stages_without_gf_opt_subprocess(self):
        graph = gf.Graph.dense(8)
        x = torch.randn(8)
        module = message_passing_domain_mlir(
            kernel=DenseSum(),
            graph=graph,
            src={"x": x},
            dst={},
            edge={},
            params={},
            kernel_name="NoSubprocessDenseSum",
        )
        with mock.patch(
            "graphforge.interop.torch.compiler_bridge.subprocess.run"
        ) as run:
            stages = lower_mlir_stages(module, gf_opt="/does/not/exist")
        run.assert_not_called()
        self.assertIn('"gf.apply"', stages.domain)
        self.assertIn('"gf_iter.traverse"', stages.iteration)
        self.assertIn('"gf_kernel.dense_launch"', stages.kernel)

    def test_optional_node_is_constructed_by_native_opbuilder(self):
        nodes, degree = 16, 2
        row_ptr = torch.arange(0, nodes * degree + 1, degree)
        col_idx = torch.arange(nodes * degree) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        u = torch.randn(nodes)
        conductivity = torch.randn(nodes * degree)
        module = message_passing_domain_mlir(
            kernel=DiffusionUpdate(),
            graph=graph,
            src={"u": u},
            dst={"u": u},
            edge={"conductivity": conductivity},
            params={"dt": 0.05},
            kernel_name="DiffusionUpdate",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())

        self.assertIn("region_kinds = array<i64: 0, 1>", stages.domain)
        self.assertEqual(stages.domain.count('"gf.yield"'), 2)
        self.assertIn("arith.mulf", stages.domain)
        self.assertIn("arith.addf", stages.domain)
        self.assertIn("region_kinds = array<i64: 0, 1>", stages.iteration)
        self.assertEqual(stages.iteration.count('"gf_iter.yield"'), 2)
        self.assertIn("region_kinds = array<i64: 0, 1>", stages.kernel)
        self.assertEqual(stages.kernel.count('"gf_kernel.yield"'), 2)

    def test_skewed_csr_exposes_physical_degree_bucket_task_ir(self):
        degrees = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64])
        row_ptr = torch.cat((torch.zeros(1, dtype=torch.int64), degrees.cumsum(0)))
        col_idx = torch.arange(int(row_ptr[-1])) % len(degrees)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=len(degrees))
        x = torch.randn(len(degrees))
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(),
            graph=graph,
            src={"x": x},
            dst={},
            edge={"weight": torch.randn(len(col_idx))},
            params={},
            kernel_name="SkewedWeightedAggregation",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())

        self.assertIsNotNone(stages.task)
        assert stages.task is not None
        self.assertIn('layout = "degree-row-worklist"', stages.task)
        self.assertIn('"gf_task.degree_reset"', stages.task)
        self.assertIn('"gf_task.degree_histogram"', stages.task)
        self.assertIn('"gf_task.degree_prefix"', stages.task)
        self.assertIn('"gf_task.degree_scatter"', stages.task)
        self.assertEqual(stages.task.count('"gf_task.degree_bucket_launch"'), 4)
        self.assertIn('partitioning = "degree-worklist"', stages.task)
        self.assertIn('"gf_storage.join"', stages.task)

        plan = gf.compiler.translate_task_bundle(stages.task)
        self.assertEqual(len(plan.invocations), 9)
        self.assertEqual(
            {item.task_kind for item in plan.invocations[:4]},
            {"degree-reset", "degree-histogram", "degree-prefix", "degree-scatter"},
        )
        self.assertEqual(plan.invocations[-1].task_kind, "join")
        self.assertEqual(plan.terminals, ("join:0",))

    def test_oversized_rows_use_compact_chunked_tail_task_ir(self):
        degrees = torch.tensor([1, 2, 4, 8, 16, 32, 128, 256])
        row_ptr = torch.cat((
            torch.zeros(1, dtype=torch.int64), degrees.cumsum(0)))
        col_idx = torch.arange(int(row_ptr[-1])) % len(degrees)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=len(degrees))
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(), graph=graph,
            src={"x": torch.randn(len(degrees))}, dst={},
            edge={"weight": torch.randn(len(col_idx))}, params={},
            kernel_name="HeavyTailWeightedAggregation",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())

        self.assertIn(
            'load_balance_plan = "degree-bucketing-required"', stages.kernel)
        assert stages.task is not None
        self.assertNotIn('layout = "row-partial-f32"', stages.task)
        self.assertIn('row_mapping = "worklist-chunked"', stages.task)
        self.assertIn(
            'load_balance_plan = "high-degree-tail-chunked"', stages.task)
        self.assertIn('"gf_task.degree_scatter"', stages.task)

        plan = gf.compiler.translate_task_bundle(stages.task)
        self.assertEqual(
            tuple(invocation.task_kind for invocation in plan.invocations),
            ("degree-histogram", "degree-reset", "degree-prefix",
             "degree-scatter", "degree-bucket", "degree-bucket", "join"),
        )
        self.assertNotIn("row_partials", {
            resource.name for resource in plan.resources
        })
        self.assertEqual(plan.invocations[-2].metadata["row_mapping"],
                         "worklist-chunked")
        self.assertIn("scf.for", plan.invocations[-2].metadata["provider_ttir"])
        bundle = plan.bind(lambda invocation: lambda **arguments: None)
        self.assertEqual(
            bundle.execution_order,
            ("degree-reset:0", "degree-histogram:0", "degree-prefix:0",
             "degree-scatter:0", "degree-bucket:0", "degree-bucket:1",
             "join:0"),
        )

    @unittest.skipUnless(_gf_translate(), "a built gf-translate is required")
    def test_dense_node_input_abi_reaches_generic_ttir(self):
        graph = gf.Graph.dense(8)
        u = torch.randn(8)
        module = message_passing_domain_mlir(
            kernel=DenseDiffusionUpdate(), graph=graph,
            src={"u": u}, dst={"u": u}, edge={},
            params={"scale": 0.25, "bias": 0.5},
            kernel_name="DenseDiffusionUpdate")
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        ttir = lower_kernel_to_ttir(
            stages.kernel, gf_translate=_gf_translate())

        for stage in (stages.domain, stages.iteration, stages.kernel):
            self.assertIn("node_input_indices = array<i64: 1, 2>", stage)
            self.assertIn("node_input_segment_sizes = array<i64: 2>", stage)
        self.assertIn("%gf_node_input1 = tt.load", ttir)
        self.assertIn("arith.addf %gf_node_input1, %gf_state#0", ttir)

    @unittest.skipUnless(
        torch.cuda.is_available() and _gf_translate(),
        "CUDA and a built gf-translate are required",
    )
    def test_public_lazy_jit_runs_udf_reducer_and_rebinds_scalar_params(self):
        nodes = 8
        graph = gf.Graph.dense(nodes, device="cuda")
        x = torch.randn(nodes, device="cuda")
        mean = DenseMean()
        actual = mean(graph=graph, src={"x": x}, dst={})
        torch.testing.assert_close(actual, x.mean().expand_as(x))
        self.assertEqual(mean.last_variant.backend, "cuda")
        self.assertIn("ttir", mean.last_variant.artifacts)

        diffusion = DenseDiffusionUpdate()
        first = diffusion(
            graph=graph, src={"u": x}, dst={"u": x},
            scale=0.25, bias=0.5)
        first_expected = x + 0.25 * (x.sum() - nodes * x) + 0.5
        torch.testing.assert_close(first, first_expected)
        second = diffusion(
            graph=graph, src={"u": x}, dst={"u": x},
            scale=0.75, bias=-0.2)
        second_expected = x + 0.75 * (x.sum() - nodes * x) - 0.2
        torch.testing.assert_close(second, second_expected)
        self.assertEqual(diffusion.cache_info["hits"], 1)
        self.assertEqual(diffusion.cache_info["variants"], 1)

        degree = 2
        row_ptr = torch.arange(
            0, nodes * degree + 1, degree, device="cuda")
        col_idx = torch.arange(nodes * degree, device="cuda") % nodes
        csr = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        csr_actual = diffusion(
            graph=csr, src={"u": x}, dst={"u": x},
            scale=0.25, bias=0.5)
        csr_flux = torch.stack([
            ((x[col_idx[row * degree:(row + 1) * degree]] - x[row]) * 0.25)
            .sum()
            for row in range(nodes)
        ])
        torch.testing.assert_close(csr_actual, x + csr_flux + 0.5)
        self.assertEqual(
            diffusion.last_variant.lowering,
            "gf-kernel-to-ttir-csr-additive-tile",
        )
        self.assertIn("block_rows=16", diffusion.ir("gf.kernel.ttir"))
        self.assertIn(
            "select-bounded-row-neighbor-tile[rows=16]",
            diffusion.last_variant.passes,
        )
        self.assertIn("padding utilization=100.0%", diffusion.explain())

    @unittest.skipUnless(
        torch.cuda.is_available() and _gf_translate(),
        "CUDA and a built gf-translate are required",
    )
    def test_static_csr_multi_message_nonadditive_reducer_compiles(self):
        nodes, degree = 128, 16
        edges = nodes * degree
        generator = torch.Generator(device="cuda").manual_seed(37)
        row_ptr = torch.arange(
            0, edges + 1, degree, device="cuda", dtype=torch.int64)
        col_idx = torch.randint(
            nodes, (edges,), device="cuda", dtype=torch.int64,
            generator=generator)
        score = torch.randn(nodes, device="cuda", generator=generator) + 1000.0
        value = torch.randn(nodes, device="cuda", generator=generator)
        kernel = ScalarOnlineAggregation()
        actual = kernel(
            graph=gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes),
            src={"score": score, "value": value}, dst={})
        score_rows = score[col_idx].reshape(nodes, degree)
        value_rows = value[col_idx].reshape(nodes, degree)
        expected = (
            torch.softmax(score_rows, dim=1) * value_rows
        ).sum(dim=1)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-csr-stable-weighted-tile",
        )
        ttir = kernel.last_variant.artifacts["ttir"]
        self.assertNotIn("scf.for", ttir)
        self.assertIn("tt.reduce", ttir)
        self.assertIn("tensor<32x16xf32>", ttir)
        self.assertIn(
            "block_rows=32 num_warps=1", kernel.ir("gf.kernel.ttir"))
        self.assertIn("math.exp", ttir)
        self.assertIn("arith.maximumf", ttir)
        self.assertIn("@gf_csr_stable_weighted_tile", ttir)
        self.assertEqual(ttir.count('"tt.reduce"'), 3)
        prepared = kernel.prepare(
            graph=gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes),
            src={"score": score, "value": value}, dst={})
        torch.testing.assert_close(
            prepared(), expected, rtol=2e-4, atol=2e-4)
        self.assertEqual(
            prepared.variant.lowering,
            "gf-kernel-to-ttir-csr-stable-weighted-tile",
        )

    @unittest.skipUnless(_gf_translate(), "a built gf-translate is required")
    def test_structured_reducer_reaches_direct_dense_ttir(self):
        nodes, lanes, width = 64, 3, 64
        graph = gf.Graph.dense(nodes)
        storage = torch.randn(lanes, nodes, width, dtype=torch.float16)
        field = storage.permute(1, 0, 2)
        module = message_passing_domain_mlir(
            kernel=StructuredDense(), graph=graph,
            src={"right": field, "payload": field},
            dst={"left": field}, edge={}, params={"scale": width**-0.5},
            kernel_name="arbitrary_user_program")
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        ttir = lower_kernel_to_ttir(
            stages.kernel, gf_translate=_gf_translate())

        self.assertIn('kind = "algebraic"', stages.domain)
        self.assertIn('"gf.yield"', stages.domain)
        self.assertIn('"gf_kernel.yield"', stages.kernel)
        self.assertIn("tt.func public @gf_dense_streaming_reduce", ttir)
        self.assertIn("%new_o = tt.dot", ttir)
        self.assertNotIn("attention", ttir.lower())

    @unittest.skipUnless(_gf_translate(), "a built gf-translate is required")
    def test_triangular_relation_boundary_reaches_dense_ttir(self):
        nodes, lanes, width = 64, 2, 64
        graph = gf.Graph.triangular(nodes)
        storage = torch.randn(lanes, nodes, width, dtype=torch.float16)
        field = storage.permute(1, 0, 2)
        module = message_passing_domain_mlir(
            kernel=StructuredDense(), graph=graph,
            src={"right": field, "payload": field},
            dst={"left": field}, edge={}, params={"scale": width**-0.5},
            kernel_name="triangular_user_program")
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        ttir = lower_kernel_to_ttir(
            stages.kernel, gf_translate=_gf_translate())

        self.assertIn('boundary = "lower_inclusive"', stages.domain)
        self.assertIn('boundary = "lower_inclusive"', stages.iteration)
        self.assertIn('boundary = "lower_inclusive"', stages.kernel)
        self.assertIn("boundary=lower_inclusive", ttir)
        self.assertIn("%causal_end", ttir)
        self.assertIn("arith.cmpi sle", ttir)
        self.assertNotIn("attention", ttir.lower())

    @unittest.skipUnless(
        torch.cuda.is_available() and _gf_translate(),
        "CUDA and a built gf-translate are required",
    )
    def test_triangular_streaming_matches_causal_gqa_sdpa(self):
        nodes, query_lanes, source_lanes, width = 128, 8, 2, 64
        generator = torch.Generator(device="cuda").manual_seed(20260813)
        query = torch.randn(
            query_lanes, nodes, width, device="cuda", dtype=torch.float16,
            generator=generator)
        key = torch.randn(
            source_lanes, nodes, width, device="cuda", dtype=torch.float16,
            generator=generator)
        value = torch.randn_like(key)
        kernel = StructuredDense()
        actual = kernel(
            graph=gf.Graph.triangular(nodes, device="cuda"),
            src={"right": key.permute(1, 0, 2),
                 "payload": value.permute(1, 0, 2)},
            dst={"left": query.permute(1, 0, 2)},
            scale=width**-0.5,
        ).permute(1, 0, 2)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True)
        torch.testing.assert_close(actual, expected, rtol=4e-3, atol=4e-3)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-dense-streaming-reduction")
        self.assertIn("%pid_source_lane", kernel.ir("gf.kernel.ttir"))
        self.assertIn(
            "%lane_group = arith.constant 4", kernel.ir("gf.kernel.ttir"))

    def test_dense_graph_reaches_dense_kernel_skeleton(self):
        graph = gf.Graph.dense(32, 16)
        x = torch.randn(32)
        module = message_passing_domain_mlir(
            kernel=DenseSum(), graph=graph,
            src={"x": x}, dst={}, edge={}, params={},
            kernel_name="DenseSum")
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())

        self.assertIn('"gf.cartesian"', stages.domain)
        self.assertIn('coordinate_hierarchy = "cartesian-product"',
                      stages.iteration)
        self.assertIn('"gf_kernel.dense_launch"', stages.kernel)
        self.assertIn('traversal = "dense-tile"', stages.kernel)
        self.assertIn('schedule_kind = "dense-query-key-tile"', stages.kernel)

    def test_udf_reducer_is_embedded_as_four_native_regions(self):
        graph = gf.Graph.dense(8)
        module = message_passing_domain_mlir(
            kernel=DenseMean(), graph=graph,
            src={"x": torch.randn(8)}, dst={}, edge={}, params={},
            kernel_name="DenseMean")
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())

        self.assertIn('state_types = [f32, f32]', stages.domain)
        self.assertIn('reducers = [@DenseMean_reducer]', stages.domain)
        self.assertEqual(stages.domain.count('"gf.reducer_yield"'), 4)
        self.assertIn('arith.divf', stages.domain)
        self.assertIn('"gf_kernel.dense_launch"', stages.kernel)

    @unittest.skipUnless(_gf_translate(), "a built gf-translate is required")
    def test_udf_tuple_reducer_lowers_by_algebra_not_python_name(self):
        graph = gf.Graph.dense(8)
        module = message_passing_domain_mlir(
            kernel=DenseMean(), graph=graph,
            src={"x": torch.randn(8)}, dst={}, edge={}, params={},
            kernel_name="UnrelatedUserKernelName")
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        ttir = lower_kernel_to_ttir(
            stages.kernel, gf_translate=_gf_translate())

        self.assertIn("tt.func public @gf_dense_scalar_reduce", ttir)
        self.assertIn("%gf_state:2 = scf.for", ttir)
        self.assertIn("arith.divf %gf_state#0, %gf_state#1", ttir)
        self.assertNotIn("DenseMean", ttir)
        self.assertNotIn("UnrelatedUserKernelName", ttir)

    def test_weighted_sum_reaches_canonical_kernel_ir(self):
        nodes, degree = 16, 2
        row_ptr = torch.arange(0, nodes * degree + 1, degree)
        col_idx = torch.arange(nodes * degree) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes)
        weight = torch.randn(nodes * degree)
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(),
            graph=graph,
            src={"x": x}, dst={}, edge={"weight": weight}, params={},
            kernel_name="WeightedAggregation",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())

        self.assertIn('kind = "algebraic"', stages.domain)
        self.assertIn('reducers = [@WeightedAggregation_reducer]', stages.domain)
        self.assertEqual(stages.domain.count('"gf.reducer_yield"'), 4)
        self.assertNotIn('reducers = ["sum"]', stages.domain)
        self.assertIn('"gf.apply"', stages.domain)
        self.assertNotIn('"gf_iter.traverse"', stages.domain)
        self.assertIn('"gf_iter.traverse"', stages.iteration)
        self.assertIn('coordinate_hierarchy = "compressed-row"', stages.iteration)
        self.assertIn('"gf_kernel.launch"', stages.kernel)
        self.assertIn('traversal = "csr-row"', stages.kernel)
        self.assertNotIn('"gf.apply"', stages.kernel)


    @unittest.skipUnless(_gf_translate(), "a built gf-translate is required")
    def test_scalar_kernel_translates_to_provider_ttir(self):
        nodes, degree = 16, 2
        row_ptr = torch.arange(0, nodes * degree + 1, degree)
        col_idx = torch.arange(nodes * degree) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes)
        weight = torch.randn(nodes * degree)
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(),
            graph=graph,
            src={"x": x}, dst={}, edge={"weight": weight}, params={},
            kernel_name="ScalarWeightedAggregation",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        ttir = lower_kernel_to_ttir(
            stages.kernel, gf_translate=_gf_translate())
        self.assertIn("tt.func public @gf_csr_weighted_sum", ttir)
        self.assertIn('%sum = "tt.reduce"', ttir)

    @unittest.skipUnless(_gf_translate(), "a built gf-translate is required")
    def test_vector_kernel_translates_to_feature_tiled_provider_ttir(self):
        nodes, degree, features = 16, 4, 16
        row_ptr = torch.arange(0, nodes * degree + 1, degree)
        col_idx = torch.arange(nodes * degree) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, features)
        weight = torch.randn(nodes * degree)
        module = message_passing_domain_mlir(
            kernel=WeightedAggregation(),
            graph=graph,
            src={"x": x}, dst={}, edge={"weight": weight}, params={},
            kernel_name="VectorWeightedAggregation",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        ttir = lower_kernel_to_ttir(
            stages.kernel, gf_translate=_gf_translate())

        self.assertIn('schedule_kind = "fixed-row-neighbor-feature"',
                      stages.kernel)
        self.assertIn("tensor<32x4x16xf32>", ttir)
        self.assertIn('%sum = "tt.reduce"', ttir)

    @unittest.skipUnless(_gf_translate(), "a built gf-translate is required")
    def test_generated_radius_reaches_direct_provider_ttir(self):
        positions = torch.rand(64, 3)
        graph = gf.Graph.radius(positions, cutoff=0.25)
        directory = graph.generated_cell_directory()
        self.assertIsNotNone(directory)
        x = torch.randn(64)
        module = message_passing_domain_mlir(
            kernel=RadiusDistanceAggregation(),
            graph=graph,
            directory=directory,
            src={"x": x}, dst={}, edge={}, params={},
            kernel_name="RadiusDistanceAggregation",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        ttir = lower_kernel_to_ttir(
            stages.kernel, gf_translate=_gf_translate())

        self.assertIn('"gf.generated_radius"', stages.domain)
        self.assertIn('coordinate_hierarchy = "generated-neighborhood"',
                      stages.iteration)
        self.assertIn('"gf_kernel.generated_launch"', stages.kernel)
        self.assertIn("tt.func public @gf_generated_radius_distance_sum", ttir)
        self.assertIn("%cell_sum = scf.for %neighbor", ttir)
        self.assertIn("%distance = math.sqrt", ttir)

    def test_message_passing_variant_exposes_canonical_stages(self):
        class WeightedAggregation(gf.MessagePassing):
            reducer = gf.sum()

            def edge(self, src, dst, edge):
                return edge.weight * src.x

        nodes, degree = 16, 2
        row_ptr = torch.arange(0, nodes * degree + 1, degree)
        col_idx = torch.arange(nodes * degree) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes)
        weight = torch.randn(nodes * degree)
        kernel = WeightedAggregation()
        with mock.patch.dict(
            os.environ, {"GRAPHFORGE_OPT": str(_gf_opt())}, clear=False):
            kernel(
                graph=graph,
                src={"x": x},
                dst={"x": x},
                edge={"weight": weight},
            )

        self.assertIn('"gf.apply"', kernel.ir("domain"))
        self.assertIn('"gf_iter.traverse"', kernel.ir("iteration"))
        self.assertIn('"gf_kernel.launch"', kernel.ir("kernel"))
        self.assertEqual(kernel.ir("gf.kernel"), kernel.ir("kernel"))
        self.assertEqual(kernel.last_variant.provider, "torch.sparse.mm")
        self.assertEqual(len(kernel.schedules), 1)
        schedule = kernel.schedules[0]
        self.assertIsInstance(schedule, gf.MachineSchedule)
        self.assertEqual(schedule.operation, "gf_kernel.launch")
        self.assertEqual(schedule.kind, "fixed-row-neighbor")
        self.assertEqual(schedule.target_contract, "provider-neutral-v1")
        self.assertEqual(schedule.pipeline_stages, 1)
        self.assertIn("relation.global.read", schedule.resources)
        self.assertIn("subgroup.neighbor-reduction", schedule.roles)
        self.assertIn("associative-reduce", schedule.instructions)
        self.assertEqual(kernel.diagnostics[0].stage, "machine-schedule")
        self.assertEqual(kernel.diagnostics[0].disposition, "accepted")
        self.assertEqual(kernel.diagnostics[1].stage, "pipeline")
        self.assertEqual(kernel.diagnostics[1].disposition, "unknown")
        self.assertIn(
            "unknown: [pipeline] no compiler-controlled asynchronous pipeline",
            kernel.explain(),
        )

    def test_native_schedule_inspector_reads_only_verified_kernel_ir(self):
        graph = gf.Graph.dense(8)
        module = message_passing_domain_mlir(
            kernel=DenseSum(), graph=graph,
            src={"x": torch.randn(8)}, dst={}, edge={}, params={},
            kernel_name="ScheduleInspection",
        )
        stages = lower_mlir_stages(module, gf_opt=_gf_opt())
        schedules = gf.compiler.schedules_from_mlir(stages.kernel)

        self.assertEqual(len(schedules), 1)
        self.assertEqual(schedules[0].operation, "gf_kernel.dense_launch")
        self.assertEqual(schedules[0].kind, "dense-query-key-tile")
        self.assertIn("vector-contraction", schedules[0].instructions)
        self.assertEqual(gf.compiler.schedules_from_mlir("module {}"), ())
        with self.assertRaisesRegex(TypeError, "non-empty MLIR"):
            gf.compiler.schedules_from_mlir("")


if __name__ == "__main__":
    unittest.main()
