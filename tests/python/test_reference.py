from __future__ import annotations

import unittest
import importlib.util
from unittest import mock

import torch

import tiga as gf


def compiler_tools_available() -> bool:
    """Use the same wheel/editable-build discovery contract as the runtime."""
    try:
        from tiga.interop.torch.compiler_bridge import (
            find_gf_opt,
            find_gf_translate,
        )

        return find_gf_opt() is not None and find_gf_translate() is not None
    except (ImportError, OSError, RuntimeError):
        return False


def ring_csr(nodes: int, degree: int, device: str = "cpu"):
    row_ptr = torch.arange(
        0, nodes * degree + 1, degree, device=device, dtype=torch.int64)
    offsets = torch.arange(1, degree + 1, device=device, dtype=torch.int64)
    col_idx = (
        torch.arange(nodes, device=device, dtype=torch.int64)[:, None] + offsets
    ) % nodes
    return row_ptr, col_idx.reshape(-1)


class Diffusion(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)

    def node(self, dst, flux, dt):
        return dst.u + dt * flux


class RadiusDisplacement(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.displacement


class DiffusionFlux(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * (src.u - dst.u)


class WeightedAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


class OnlineAttention(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge):
        del dst
        return self.reducer(edge.score, src.value)


class DenseAttention(gf.MessagePassing):
    reducer = gf.online_softmax()

    def edge(self, src, dst, edge, scale):
        del edge
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


class TilePrunedDenseAttention(gf.MessagePassing):
    reducer = gf.online_softmax(block_prune_threshold=0.125)

    def edge(self, src, dst, edge, scale):
        del edge
        score = (src.key * dst.query).sum(dim=-1) * scale
        return self.reducer(score, src.value)


class RadiusDistanceAggregation(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


class DifferentTypeRadius(gf.RadiusGraph):
    def metric(self, src, dst, edge, scale):
        del src, dst
        return torch.linalg.vector_norm(edge.displacement * scale, dim=-1)

    def select(self, src, dst, edge):
        del edge
        return src.particle_type != dst.particle_type


class ReferenceTest(unittest.TestCase):
    def test_torch_coo_import_rejects_negative_and_explicit_overflow(self):
        with self.assertRaisesRegex(ValueError, "out-of-range source"):
            gf.Graph.from_coo(
                torch.tensor([-1]), torch.tensor([0]),
                num_src=1, num_dst=1)
        with self.assertRaisesRegex(ValueError, "out-of-range destination"):
            gf.Graph.from_coo(
                torch.tensor([0]), torch.tensor([2]),
                num_src=1, num_dst=2)

    def test_compiler_derived_source_vjp_reuses_transpose_relation(self):
        from tiga.interop.torch.relation_vjp import compile_source_vjp

        row_ptr, col_idx = ring_csr(8, 2)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=8)
        weight = torch.linspace(0.25, 1.0, col_idx.numel())
        x = torch.linspace(-1.0, 1.0, 8)
        cotangent = torch.linspace(1.0, 2.0, 8)
        kernel = WeightedAggregation()

        generated = compile_source_vjp(
            kernel, graph=graph, src={"x": x}, edge={"weight": weight})
        actual = generated(cotangent)
        source = col_idx
        destination = torch.repeat_interleave(torch.arange(8), 2)
        expected = torch.zeros_like(x)
        expected.index_add_(0, source, weight * cotangent[destination])

        torch.testing.assert_close(actual, expected)
        self.assertEqual(generated.transform,
                         "transpose-relation-weighted-sum")
        self.assertIn(generated.program.last_variant.lowering, {
            "dispatch-native-sparse-mm",
            "gf-kernel-to-ttir-fixed-csr-weighted-sum",
            "gf-kernel-to-ttir-bounded-ragged-weighted-sum",
        })

    def test_degree_statistics_participate_in_planning_key(self):
        # Both graphs have the same schema and min/max degree but a different
        # edge sum, so padding/load-balance plans must not share a cache key.
        first_degrees = torch.tensor([0, 1, 3, 4], dtype=torch.int64)
        second_degrees = torch.tensor([0, 2, 3, 4], dtype=torch.int64)

        def graph_from_degrees(degrees):
            row_ptr = torch.empty(5, dtype=torch.int64)
            row_ptr[0] = 0
            torch.cumsum(degrees, 0, out=row_ptr[1:])
            col_idx = torch.arange(int(row_ptr[-1]), dtype=torch.int64) % 4
            return gf.Graph.from_csr(row_ptr, col_idx, num_src=4)

        first = graph_from_degrees(first_degrees)
        second = graph_from_degrees(second_degrees)
        self.assertEqual(first.degree_statistics(), (0, 4, 8))
        self.assertEqual(second.degree_statistics(), (0, 4, 9))
        self.assertNotEqual(first.planning_key(), second.planning_key())

    def test_source_index_span_refines_physical_planning_key(self):
        nodes = 64
        degree = 4
        row_ptr = torch.arange(
            0, (nodes + 1) * degree, degree, dtype=torch.int64)
        destinations = torch.repeat_interleave(torch.arange(nodes), degree)
        local = (destinations + torch.arange(degree).repeat(nodes)) % nodes
        remote = (local + nodes // 2) % nodes
        local_graph = gf.Graph.from_csr(row_ptr, local, num_src=nodes)
        remote_graph = gf.Graph.from_csr(row_ptr, remote, num_src=nodes)

        self.assertLess(local_graph.source_index_span_ratio(), 0.05)
        self.assertGreater(remote_graph.source_index_span_ratio(), 0.25)
        self.assertNotEqual(local_graph.planning_key(), remote_graph.planning_key())

    def test_graph_halo_is_declarative_and_preserves_graph_type(self):
        row_ptr, col_idx = ring_csr(8, 2)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=8)
        mesh = gf.DeviceMesh("cuda", (2, 4), names=("x", "y"))

        distributed = graph.halo(
            mesh,
            partition=gf.ByDestination(mesh_axis="x", balance="edges"),
            depth="auto",
        )

        self.assertIs(type(distributed), gf.Graph)
        self.assertFalse(graph.is_distributed)
        self.assertTrue(distributed.is_distributed)
        self.assertEqual(distributed.placement.mesh.size, 8)
        self.assertEqual(distributed.placement.halo_depth, "auto")
        self.assertNotEqual(graph.planning_key(), distributed.planning_key())
        self.assertIn("halo=auto", distributed.explain())

    def test_graph_halo_requires_an_active_distributed_runtime(self):
        row_ptr, col_idx = ring_csr(8, 2)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=8).halo(
            gf.DeviceMesh("cpu", 2), depth=1)
        x = torch.randn(8)
        weight = torch.randn(16)

        kernel = WeightedAggregation()
        with self.assertRaisesRegex(NotImplementedError, "active DistributedRuntime"):
            kernel(
                graph=graph,
                src={"x": x},
                dst={"x": x},
                edge={"weight": weight},
            )
        task = kernel.ir("task")
        self.assertIn('"gf_task.halo_pack"', task)
        self.assertIn('"gf_task.halo_exchange"', task)
        self.assertIn('"gf_task.halo_unpack"', task)
        self.assertLess(task.index('task_kind = "interior"'),
                        task.index('task_kind = "boundary"'))

    def test_graph_halo_validates_mesh_axis_and_depth(self):
        graph = gf.Graph.dense(4)
        mesh = gf.DeviceMesh("cuda", (2, 2), names=("x", "y"))
        with self.assertRaisesRegex(ValueError, "outside"):
            graph.halo(mesh, partition=gf.ByDestination(mesh_axis=2))
        with self.assertRaisesRegex(ValueError, "halo depth"):
            graph.halo(mesh, depth=-1)

    def test_exact_owner_ghost_halo_map_and_byte_sizing(self):
        # Rank 0 owns destinations [0, 2); their remote sources are 2,3,5.
        row_ptr = [0, 3, 5, 7, 9, 11, 13]
        col_idx = [0, 2, 5, 1, 3, 2, 4, 0, 3, 1, 4, 2, 5]
        halo = gf.derive_halo_map(
            row_ptr,
            col_idx,
            num_entities=6,
            world_size=3,
            rank=0,
            peer_requests=((), (0,), (1, 0)),
        )
        self.assertEqual((halo.owned_begin, halo.owned_end), (0, 2))
        self.assertEqual(halo.ghost_ids, (2, 3, 5))
        self.assertEqual(halo.receive_from, ((1, (2, 3)), (2, (5,))))
        self.assertEqual(halo.send_to, ((1, (0,)), (2, (0, 1))))
        self.assertEqual(halo.bytes_for(4, 16), 3 * 4 * 16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dense_graph_normalizes_unindexed_cuda_device(self):
        graph = gf.Graph.dense(4, device="cuda")
        self.assertEqual(
            graph.device,
            gf.Device.parse(f"cuda:{torch.cuda.current_device()}"),
        )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and importlib.util.find_spec("triton")
        and compiler_tools_available(),
        "CUDA, Triton, gf-opt and gf-translate are required",
    )
    def test_dense_structured_reducer_uses_direct_compiler_ttir(self):
        nodes, heads, width = 512, 2, 64
        query_storage = torch.randn(
            heads, nodes, width, device="cuda", dtype=torch.float16)
        key_storage = torch.randn_like(query_storage)
        value_storage = torch.randn_like(query_storage)
        query = query_storage.permute(1, 0, 2)
        key = key_storage.permute(1, 0, 2)
        value = value_storage.permute(1, 0, 2)
        graph = gf.Graph.dense(nodes, device=query.device)
        kernel = DenseAttention()

        actual = kernel(
            graph=graph,
            src={"key": key, "value": value},
            dst={"query": query},
            scale=width**-0.5,
        )
        expected = torch.nn.functional.scaled_dot_product_attention(
            query_storage[None], key_storage[None], value_storage[None]
        )[0].permute(1, 0, 2)

        torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-dense-streaming-reduction")
        self.assertIn('"gf_kernel.dense_launch"', kernel.ir("kernel"))
        self.assertIn("tt.func public @gf_dense_streaming_reduce",
                      kernel.ir("kernel_ttir"))
        self.assertGreater(len(kernel.code("ptx")), 1000)
        self.assertEqual(graph.num_edges, nodes * nodes)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and importlib.util.find_spec("triton")
        and compiler_tools_available(),
        "CUDA, Triton, gf-opt and gf-translate are required",
    )
    def test_dense_block_pruning_is_reducer_ir_and_dynamic_ttir(self):
        nodes, heads, width = 512, 2, 64
        query_storage = torch.randn(
            heads, nodes, width, device="cuda", dtype=torch.float16)
        key_storage = torch.randn_like(query_storage)
        value_storage = torch.randn_like(query_storage)
        query, key, value = (
            item.permute(1, 0, 2)
            for item in (query_storage, key_storage, value_storage)
        )
        kernel = TilePrunedDenseAttention()
        actual = kernel(
            graph=gf.Graph.dense(nodes, device="cuda"),
            src={"key": key, "value": value}, dst={"query": query},
            scale=width**-0.5,
        )
        self.assertTrue(torch.isfinite(actual).all())
        self.assertIn("block_prune_threshold", kernel.ir("domain"))
        source = kernel.ir("kernel_ttir")
        self.assertIn("tiga.reducer block_prune=", source)
        self.assertIn("scf.if", source)
        self.assertIn("%old_block_m", source)

    def test_block_pruning_refuses_silent_exact_cpu_fallback(self):
        with self.assertRaisesRegex(NotImplementedError, "generated CUDA"):
            TilePrunedDenseAttention()(
                graph=gf.Graph.dense(4),
                src={
                    "key": torch.randn(4, 16),
                    "value": torch.randn(4, 16),
                },
                dst={"query": torch.randn(4, 16)},
                scale=0.25,
            )

    def test_dense_graph_is_implicit_and_attention_uses_same_semantics(self):
        sources, destinations, width = 4, 3, 5
        graph = gf.Graph.dense(sources, destinations)
        query = torch.randn(destinations, width)
        key = torch.randn(sources, width)
        value = torch.randn(sources, 2)
        scale = width**-0.5

        actual = DenseAttention()(
            graph=graph,
            src={"key": key, "value": value},
            dst={"query": query},
            scale=scale,
        )
        score = query @ key.T * scale
        expected = torch.softmax(score, dim=-1) @ value

        torch.testing.assert_close(actual, expected)
        self.assertEqual(graph.num_edges, sources * destinations)
        self.assertIn("realization=implicit_dense", graph.explain())

    def test_triangular_graph_is_implicit_and_reference_is_lower_inclusive(self):
        graph = gf.Graph.triangular(4)
        self.assertEqual(graph.num_edges, 10)
        self.assertEqual(graph.degree_statistics(), (1, 4, 10))
        self.assertEqual(graph.degree_histogram(), (0, 1, 1, 1, 1))
        row_ptr, col_idx = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [0, 1, 3, 6, 10])
        self.assertEqual(
            col_idx.tolist(), [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
        self.assertIn("boundary=lower_inclusive", graph.explain())

    def test_ndata_binds_both_endpoint_roles_on_homogeneous_graphs(self):
        row_ptr = torch.tensor([0, 2, 4, 6, 8], dtype=torch.int64)
        col_idx = torch.tensor(
            [1, 3, 0, 2, 1, 3, 0, 2], dtype=torch.int64)  # ring, both ways
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=4)
        u = torch.tensor([0.0, 1.0, 0.5, -0.5])
        weight = torch.ones(8)

        via_ndata = Diffusion()(
            graph=graph, ndata={"u": u}, edge={"weight": weight}, dt=0.1)
        via_roles = Diffusion()(
            graph=graph, src={"u": u}, dst={"u": u},
            edge={"weight": weight}, dt=0.1)
        torch.testing.assert_close(via_ndata, via_roles)

        with self.assertRaisesRegex(TypeError, "cannot be combined"):
            Diffusion()(graph=graph, ndata={"u": u}, dst={}, edge={})
        bipartite = gf.Graph.from_csr(
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 2], dtype=torch.int64),
            num_src=3,
        )
        with self.assertRaisesRegex(ValueError, "homogeneous"):
            Diffusion()(graph=bipartite, ndata={"u": u}, edge={})

    def test_regular_graph_has_fixed_degree_and_modular_sources(self):
        graph = gf.Graph.regular(4, 2)
        self.assertEqual(graph.num_edges, 8)
        self.assertEqual(graph.degree_statistics(), (2, 2, 8))
        row_ptr, col_idx = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(col_idx.tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "num_nodes"):
            gf.Graph.regular(0, 2)
        with self.assertRaisesRegex(ValueError, "degree"):
            gf.Graph.regular(4, -1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_regular_graph_builds_index_tensors_on_cuda(self):
        graph = gf.Graph.regular(8, 3, device="cuda")
        row_ptr, col_idx = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [3 * i for i in range(9)])
        self.assertEqual(col_idx.tolist(), [e % 8 for e in range(24)])

    def test_large_dense_graph_refuses_csr_materialization(self):
        graph = gf.Graph.dense(4096, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "tiled consumer"):
            graph.resolve_csr()

    def test_online_softmax_infers_heads_and_value_width(self):
        # Row 1 is deliberately empty; its finalized reducer identity is zero.
        row_ptr = torch.tensor([0, 2, 2, 5], dtype=torch.int64)
        col_idx = torch.tensor([0, 2, 1, 2, 3], dtype=torch.int64)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=4)
        heads, width = 2, 3
        value = torch.arange(
            4 * heads * width, dtype=torch.float32).reshape(4, heads, width)
        score = torch.tensor([
            [1000.0, -1000.0], [999.0, -999.0],
            [0.2, 0.5], [0.7, -0.5], [-0.1, 1.5],
        ])

        actual = OnlineAttention()(
            graph=graph,
            src={"value": value},
            dst={},
            edge={"score": score},
        )
        expected = torch.zeros(3, heads, width)
        for row in (0, 2):
            begin, end = int(row_ptr[row]), int(row_ptr[row + 1])
            probabilities = torch.softmax(score[begin:end], dim=0)
            expected[row] = torch.sum(
                probabilities[..., None] * value[col_idx[begin:end]], dim=0)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (3, heads, width))
        self.assertTrue(torch.isfinite(actual).all())

    def test_online_softmax_rejects_misaligned_lane_shape(self):
        reducer = gf.online_softmax()
        item = reducer(torch.randn(4, 2), torch.randn(4, 3, 5))
        from tiga.interop.torch.reducer_oracle import reduce_online_softmax
        with self.assertRaisesRegex(ValueError, "score lane dimensions"):
            reduce_online_softmax(
                reducer, item, torch.tensor([0, 0, 1, 1]), num_dst=2)

    def test_static_diffusion_and_lazy_variant(self):
        nodes, degree = 19, 4
        row_ptr, col_idx = ring_csr(nodes, degree)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes, validate="full")
        u = torch.randn(nodes)
        weight = torch.randn(nodes * degree)
        kernel = Diffusion()
        actual = kernel(
            graph=graph,
            src={"u": u},
            dst={"u": u},
            edge={"weight": weight},
            dt=0.125,
        )
        dst_index = torch.arange(nodes).repeat_interleave(degree)
        flux = torch.zeros_like(u)
        flux.index_add_(0, dst_index, weight * (u[col_idx] - u[dst_index]))
        torch.testing.assert_close(actual, u + 0.125 * flux)
        self.assertEqual(len(kernel.variants), 1)
        self.assertIn("apply @Diffusion", kernel.ir())
        self.assertIn("reference evaluator", kernel.explain())
        kernel(
            graph=graph,
            src={"u": u},
            dst={"u": u},
            edge={"weight": weight},
            dt=0.5,
        )
        self.assertEqual(len(kernel.variants), 1, "runtime scalar values must not specialize")

    def test_dynamic_radius_implicit_geometry(self):
        positions = torch.tensor([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
        graph = gf.Graph.radius(positions, 0.75)
        actual = RadiusDisplacement()(
            graph=graph,
            src={"p": positions},
            dst={"p": positions},
        )
        expected = torch.tensor([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.0]])
        torch.testing.assert_close(actual, expected)
        self.assertIn("lifecycle=dynamic", graph.explain())
        self.assertEqual(graph.build_info["builder"], "uniform_cell_list")

    def test_dynamic_radius_reuses_equivalent_physical_snapshot(self):
        positions = torch.tensor(
            [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        first_graph = gf.Graph.radius(positions, 0.75)
        kernel = RadiusDistanceAggregation()
        first = kernel(
            graph=first_graph, src={"x": x}, dst={"x": x})

        equivalent_graph = gf.Graph.radius(positions, 0.75)
        with mock.patch.object(
            equivalent_graph,
            "resolve_csr",
            side_effect=AssertionError("equivalent snapshot was rebuilt"),
        ):
            equivalent = kernel(
                graph=equivalent_graph, src={"x": x}, dst={"x": x})
        torch.testing.assert_close(equivalent, first)

        # Versioning is part of the physical-snapshot guard: after mutation
        # the executable must resolve the changed relation instead of reusing
        # stale CSR/distance state.
        positions.add_(2.0)
        changed_graph = gf.Graph.radius(positions, 0.75)
        changed = kernel(
            graph=changed_graph, src={"x": x}, dst={"x": x})
        expected = kernel.reference(
            graph=changed_graph, src={"x": x}, dst={"x": x})
        torch.testing.assert_close(changed, expected)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and importlib.util.find_spec("triton")
        and compiler_tools_available(),
        "CUDA, Triton, gf-opt and gf-translate are required",
    )
    def test_dynamic_radius_implicit_distance_uses_compiler_or_reference(self):
        nodes = 2048
        generator = torch.Generator(device="cuda").manual_seed(23)
        positions = torch.rand(nodes, 3, device="cuda", generator=generator)
        # About eight interior neighbors in a unit cube.  This keeps the
        # bounded-ragged proof below the current max-degree-64 contract.
        cutoff = (8.0 / (nodes * (4.0 * torch.pi / 3.0))) ** (1.0 / 3.0)
        graph = gf.Graph.radius(positions, cutoff)
        x = torch.randn(nodes, device="cuda", generator=generator)
        kernel = RadiusDistanceAggregation()

        actual = kernel(graph=graph, src={"x": x}, dst={"x": x})
        expected = kernel.reference(
            graph=graph, src={"x": x}, dst={"x": x})

        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        self.assertTrue(kernel.last_variant.provider.startswith("triton-nvidia@"))
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-generated-radius-distance-sum",
        )
        self.assertIsNone(graph.num_edges)
        self.assertIn("cuda:120:warp32", kernel.last_variant.provider_key)
        self.assertEqual(graph.build_info["builder"], "uniform_cell_list")
        self.assertIn("elide-implicit-edge-geometry", kernel.explain())

    def test_dynamic_radius_custom_metric(self):
        positions = torch.tensor(
            [[0.0, 0.0], [0.6, 0.0], [0.0, 0.6]], dtype=torch.float32)

        def anisotropic_metric(src, dst, edge):
            del src, dst
            scaled = edge.displacement * torch.tensor([2.0, 1.0])
            return torch.linalg.vector_norm(scaled, dim=-1)

        graph = gf.Graph.radius(
            positions, cutoff=0.75, metric=anisotropic_metric)
        row_ptr, col_idx = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [0, 1, 1, 2])
        self.assertEqual(col_idx.tolist(), [2, 0])
        actual = RadiusDisplacement()(
            graph=graph,
            src={"p": positions},
            dst={"p": positions},
        )
        expected = torch.tensor([[0.0, 0.6], [0.0, 0.0], [0.0, -0.6]])
        torch.testing.assert_close(actual, expected)
        self.assertIn("metric=custom", graph.explain())
        self.assertIn("broad_phase=all-pairs-only", graph.explain())
        self.assertEqual(graph.build_info["builder"], "all_pairs")

    def test_dynamic_radius_select_udf_and_endpoint_fields(self):
        positions = torch.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]], dtype=torch.float32)
        particle_type = torch.tensor([0, 0, 1])

        def cross_particle_type(src, dst, edge):
            # select is an additional filter; the cutoff remains mandatory.
            return (src.particle_type != dst.particle_type) & (edge.distance > 0)

        graph = gf.Graph.radius(
            positions,
            cutoff=0.5,
            fields={"particle_type": particle_type},
            select=cross_particle_type,
        )
        row_ptr, col_idx = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [0, 0, 1, 2])
        self.assertEqual(col_idx.tolist(), [2, 1])
        self.assertIn("select=custom", graph.explain())
        self.assertIn("broad_phase=euclidean-cell-list-eligible", graph.explain())
        self.assertEqual(graph.build_info["builder"], "uniform_cell_list")

    def test_cell_list_matches_all_pairs_radius(self):
        generator = torch.Generator().manual_seed(17)
        positions = torch.rand(97, 3, generator=generator)
        cutoff = 0.23
        graph = gf.Graph.radius(positions, cutoff)
        row_ptr, col_idx = graph.resolve_csr()
        destination = graph.destination_index(row_ptr)
        actual = torch.zeros(97, 97, dtype=torch.bool)
        actual[destination, col_idx] = True
        expected = torch.cdist(positions, positions) <= cutoff
        expected.fill_diagonal_(False)
        torch.testing.assert_close(actual, expected)
        self.assertLess(
            graph.build_info["candidate_pairs"], positions.shape[0] ** 2)

    def test_periodic_orthogonal_and_skew_radius_match_minimum_image(self):
        generator = torch.Generator().manual_seed(29)
        for lattice in (
            torch.tensor([1.0, 1.0]),
            torch.tensor([[1.0, 0.0], [0.45, 0.85]]),
        ):
            matrix = torch.diag(lattice) if lattice.ndim == 1 else lattice
            positions = torch.rand(173, 2, generator=generator) @ matrix
            cutoff = 0.14
            graph = gf.Graph.radius(
                positions, cutoff, periodic=lattice
            )
            row_ptr, col_idx = graph.resolve_csr()
            destination = graph.destination_index(row_ptr)
            actual = torch.zeros(173, 173, dtype=torch.bool)
            actual[destination, col_idx] = True

            displacement = positions[None, :, :] - positions[:, None, :]
            fractional = displacement @ torch.linalg.inv(matrix)
            fractional -= torch.round(fractional)
            distance = torch.linalg.vector_norm(fractional @ matrix, dim=-1)
            expected = distance <= cutoff
            expected.fill_diagonal_(False)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(graph.build_info["builder"], "uniform_cell_list")
            self.assertLess(graph.build_info["candidate_pairs"], 173 ** 2)
            self.assertIn("periodic=", graph.explain())

    def test_generated_cell_directory_reuses_immutable_snapshot_and_invalidates(self):
        from tiga.interop.torch.graph import from_native

        positions = torch.rand(64, 3)
        graph = gf.Graph.radius(positions, 0.25)
        execution_graph = from_native(graph)
        first = execution_graph.generated_cell_directory()
        second = execution_graph.generated_cell_directory()
        self.assertIs(first, second)

        positions.add_(0.01)
        rebuilt = execution_graph.generated_cell_directory()
        self.assertIsNot(rebuilt, first)

    def test_dynamic_radius_udf_contract_errors(self):
        positions = torch.randn(3, 2)
        with self.assertRaisesRegex(ValueError, "reserved 'position'"):
            gf.Graph.radius(
                positions, 1.0, fields={"position": positions})

        graph = gf.Graph.radius(
            positions, 1.0,
            select=lambda src, dst, edge: edge.distance,
        )
        with self.assertRaisesRegex(TypeError, "boolean"):
            graph.resolve_csr()

    def test_class_based_radius_builder_and_lambda_shorthand(self):
        positions = torch.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.8, 0.0]], dtype=torch.float32)
        particle_type = torch.tensor([0, 0, 1])
        scale = torch.tensor([1.0, 1.0])
        builder = DifferentTypeRadius()
        class_graph = builder(
            positions=positions,
            cutoff=0.5,
            fields={"particle_type": particle_type},
            scale=scale,
        )
        row_ptr, col_idx = class_graph.resolve_csr()
        class_keys = torch.sort(
            class_graph.destination_index(row_ptr) * positions.shape[0] + col_idx
        ).values

        lambda_graph = gf.Graph.radius(
            positions,
            cutoff=0.5,
            fields={"particle_type": particle_type},
            metric=lambda src, dst, edge: torch.linalg.vector_norm(
                edge.displacement * scale, dim=-1),
            select=lambda src, dst, edge: src.particle_type != dst.particle_type,
        )
        row_ptr, col_idx = lambda_graph.resolve_csr()
        lambda_keys = torch.sort(
            lambda_graph.destination_index(row_ptr) * positions.shape[0] + col_idx
        ).values
        torch.testing.assert_close(class_keys, lambda_keys)
        self.assertIn("DifferentTypeRadius", builder.explain())

    def test_empty_static_graph(self):
        graph = gf.Graph.from_csr(
            torch.zeros(6, dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            num_src=5,
            validate="full",
        )
        u = torch.randn(5)
        out = Diffusion()(
            graph=graph,
            src={"u": u},
            dst={"u": u},
            edge={"weight": torch.empty(0)},
            dt=0.1,
        )
        torch.testing.assert_close(out, u)

    def test_linear_weighted_sum_dispatches_native_sparse_on_cpu(self):
        nodes, degree, features = 23, 3, 5
        row_ptr, col_idx = ring_csr(nodes, degree)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, features)
        weight = torch.randn(nodes * degree, 1)
        kernel = WeightedAggregation()
        actual = kernel(
            graph=graph,
            src={"x": x},
            dst={"x": x},
            edge={"weight": weight},
        )
        expected = kernel.reference(
            graph=graph,
            src={"x": x},
            dst={"x": x},
            edge={"weight": weight},
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(kernel.last_variant.provider, "torch.sparse.mm")
        self.assertIn("recognize-linear-weighted-sum", kernel.explain())
        self.assertTrue(kernel.diagnostics)
        finding = next(
            item for item in kernel.diagnostics if item.stage == "planning"
        )
        self.assertIsInstance(finding, gf.AnalysisFinding)
        self.assertEqual(finding.stage, "planning")
        self.assertEqual(finding.disposition, "remark")
        self.assertEqual(kernel.diagnostics[0].stage, "machine-schedule")
        with self.assertRaisesRegex(ValueError, "disposition"):
            gf.AnalysisFinding("planning", "maybe", "invalid")
        with self.assertRaisesRegex(ValueError, "positive"):
            gf.MachineSchedule(
                operation="gf_kernel.launch",
                source_location="loc(unknown)",
                kind="invalid",
                block_rows=0,
                block_neighbors=1,
                num_warps=1,
                pipeline_stages=1,
                target_contract="provider-neutral-v1",
                resources=("input.global.read",),
                roles=("lane.scalar",),
                handoffs=("load.global-to-register",),
                instructions=("scalar-control-flow",),
            )

    def test_static_auto_warm_call_uses_guarded_executable(self):
        nodes, degree, features = 17, 3, 4
        row_ptr, col_idx = ring_csr(nodes, degree)
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        weight = torch.randn(nodes * degree, 1)
        kernel = WeightedAggregation()
        first_x = torch.randn(nodes, features)
        kernel(
            graph=graph,
            src={"x": first_x},
            dst={"x": first_x},
            edge={"weight": weight},
        )

        def fail_if_recaptured(params):
            raise AssertionError("warm guarded executable unexpectedly recaptured")

        kernel._weighted_sum_pattern = fail_if_recaptured
        second_x = torch.randn_like(first_x)
        actual = kernel(
            graph=graph,
            src={"x": second_x},
            dst={"x": second_x},
            edge={"weight": weight},
        )
        expected = torch.sparse.mm(graph.sparse_csr(weight), second_x)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(kernel.cache_info, {"hits": 1, "misses": 1, "variants": 1})
        self.assertIn("executable cache: hits=1, misses=1", kernel.explain())

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton"),
        "CUDA and Triton are required",
    )
    def test_vector_weighted_sum_dispatches_compiler_feature_tile(self):
        nodes, degree, features = 1024, 8, 16
        row_ptr, col_idx = ring_csr(nodes, degree, "cuda")
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, features, device="cuda")
        weight = torch.randn(nodes * degree, 1, device="cuda")
        kernel = WeightedAggregation()
        actual = kernel(
            graph=graph,
            src={"x": x},
            dst={"x": x},
            edge={"weight": weight},
        )
        expected = kernel.reference(
            graph=graph,
            src={"x": x},
            dst={"x": x},
            edge={"weight": weight},
        )
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        retained_actual = actual.clone()
        self.assertTrue(kernel.last_variant.provider.startswith("triton-"))
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-fixed-csr-weighted-sum",
        )
        self.assertIn("row-neighbor-feature tile", kernel.explain())
        next_x = torch.randn_like(x)
        warm_actual = kernel(
            graph=graph,
            src={"x": next_x},
            dst={"x": next_x},
            edge={"weight": weight},
        )
        warm_expected = kernel.reference(
            graph=graph,
            src={"x": next_x},
            dst={"x": next_x},
            edge={"weight": weight},
        )
        torch.testing.assert_close(
            warm_actual, warm_expected, rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(actual, retained_actual)

        prepared = kernel.prepare(
            graph=graph,
            src={"x": next_x},
            dst={"x": next_x},
            edge={"weight": weight},
        )
        torch.testing.assert_close(
            prepared(), warm_expected, rtol=3e-4, atol=3e-4
        )

        # The prepared sparse view aliases the captured values buffer, so a
        # value-only mutation must be visible without rebuilding topology.
        weight.mul_(0.5)
        mutated = kernel(
            graph=graph,
            src={"x": next_x},
            dst={"x": next_x},
            edge={"weight": weight},
        )
        mutated_expected = torch.sparse.mm(graph.sparse_csr(weight), next_x)
        torch.testing.assert_close(
            mutated, mutated_expected, rtol=3e-4, atol=3e-4
        )

        # Rebinding to another same-schema values Tensor remains legal and
        # takes the guarded slow path once to bind the matching sparse view.
        replacement_weight = torch.randn_like(weight)
        rebound = kernel(
            graph=graph,
            src={"x": next_x},
            dst={"x": next_x},
            edge={"weight": replacement_weight},
        )
        rebound_expected = torch.sparse.mm(
            graph.sparse_csr(replacement_weight), next_x
        )
        torch.testing.assert_close(
            rebound, rebound_expected, rtol=3e-4, atol=3e-4
        )

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton"),
        "CUDA and Triton are required",
    )
    def test_vector_ragged_weighted_sum_uses_compiler_feature_tile(self):
        nodes, features = 1024, 16
        degrees = torch.arange(nodes, device="cuda", dtype=torch.int64) % 17
        row_ptr = torch.empty(nodes + 1, device="cuda", dtype=torch.int64)
        row_ptr[0] = 0
        torch.cumsum(degrees, dim=0, out=row_ptr[1:])
        edges = int(row_ptr[-1].item())
        col_idx = torch.arange(edges, device="cuda", dtype=torch.int64) % nodes
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        x = torch.randn(nodes, features, device="cuda")
        weight = torch.randn(edges, 1, device="cuda")
        kernel = WeightedAggregation()
        actual = kernel(
            graph=graph,
            src={"x": x},
            dst={"x": x},
            edge={"weight": weight},
        )
        expected = kernel.reference(
            graph=graph,
            src={"x": x},
            dst={"x": x},
            edge={"weight": weight},
        )
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        self.assertTrue(kernel.last_variant.provider.startswith("triton-"))
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-bounded-ragged-weighted-sum",
        )
        self.assertIn(
            "bounded-ragged-row-neighbor-feature", kernel.ir("kernel")
        )
        self.assertIn("ptx", kernel.last_variant.artifacts)

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton"),
        "CUDA and Triton are required",
    )
    def test_diffusion_uses_generic_compiler_path(self):
        nodes, degree = 1024, 8
        row_ptr, col_idx = ring_csr(nodes, degree, "cuda")
        graph = gf.Graph.from_csr(row_ptr, col_idx, num_src=nodes)
        u = torch.randn(nodes, device="cuda")
        weight = torch.randn(nodes * degree, device="cuda")
        kernel = DiffusionFlux()
        actual = kernel(
            graph=graph, src={"u": u}, dst={"u": u}, edge={"weight": weight})
        expected = kernel.reference(
            graph=graph, src={"u": u}, dst={"u": u}, edge={"weight": weight})
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-csr-additive-tile",
        )
        self.assertIn("block_rows=16", kernel.ir("gf.kernel.ttir"))

        different_destination = torch.randn_like(u)
        deoptimized = kernel(
            graph=graph,
            src={"u": u},
            dst={"u": different_destination},
            edge={"weight": weight},
        )
        deoptimized_expected = kernel.reference(
            graph=graph,
            src={"u": u},
            dst={"u": different_destination},
            edge={"weight": weight},
        )
        torch.testing.assert_close(deoptimized, deoptimized_expected)
        self.assertEqual(
            kernel.last_variant.lowering,
            "gf-kernel-to-ttir-csr-additive-tile",
        )

    def test_exact_knn_is_procedural_and_materializes_matched_neighbors(self):
        positions = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 4.0]
        ])
        graph = gf.Graph.knn(positions, 2)
        self.assertEqual(graph.schema.origin, "geometric_knn")
        self.assertEqual(graph.schema.realization, "procedural_knn")
        self.assertIsNone(graph.num_edges)
        row_ptr, col_idx = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [0, 2, 4, 6, 8])
        expected = torch.cdist(positions, positions)
        expected.fill_diagonal_(float("inf"))
        expected_idx = expected.topk(
            2, largest=False, sorted=True, dim=1).indices.reshape(-1)
        self.assertEqual(col_idx.tolist(), expected_idx.tolist())
        self.assertEqual(graph.build_info["accepted_edges"], 8)
        first_row_ptr = row_ptr
        positions.add_(1.0)
        rebuilt_row_ptr, rebuilt_col_idx = graph.resolve_csr()
        self.assertIs(rebuilt_row_ptr, first_row_ptr)
        self.assertIsNot(rebuilt_col_idx, col_idx)
        self.assertIn("k=2", graph.explain())

        with self.assertRaisesRegex(ValueError, "k must be"):
            gf.Graph.knn(positions, 4)

    def test_bipartite_knn_and_transpose_form_clustering_relations(self):
        queries = torch.tensor([
            [0.0, 0.0], [4.0, 0.0], [0.5, 0.0], [3.5, 0.0]
        ])
        candidates = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
        graph = gf.Graph.knn(queries, 1, candidates=candidates)

        self.assertEqual((graph.schema.num_src, graph.schema.num_dst), (2, 4))
        row_ptr, assignment = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(assignment.tolist(), [0, 1, 0, 1])
        geometry = graph.implicit_edge_fields(row_ptr, assignment)
        torch.testing.assert_close(
            geometry["distance"], torch.tensor([0.0, 0.0, 0.5, 0.5]))
        self.assertIn("search=bipartite", graph.explain())

        reverse = graph.transpose()
        self.assertEqual((reverse.schema.num_src, reverse.schema.num_dst), (4, 2))
        reverse_rows, reverse_columns = reverse.resolve_csr()
        self.assertEqual(reverse_rows.tolist(), [0, 2, 4])
        self.assertEqual(reverse_columns.tolist(), [0, 2, 1, 3])

        first_rows = row_ptr
        candidates[0, 0] = 10.0
        rebuilt_rows, rebuilt_assignment = graph.resolve_csr()
        self.assertIs(rebuilt_rows, first_rows)
        self.assertNotEqual(rebuilt_assignment.tolist(), assignment.tolist())

        with self.assertRaisesRegex(ValueError, "only defined for self-kNN"):
            gf.Graph.knn(
                queries, 1, candidates=candidates, exclude_self=True)


if __name__ == "__main__":
    unittest.main()
