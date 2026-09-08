from __future__ import annotations

import os
import ctypes
from pathlib import Path
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

import tiga as gf
from tiga.compiler.toolchain import find_gf_translate


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


class TensorRuntimeTest(unittest.TestCase):
    def test_scatter_rows_native_forward_vjp_and_lowering(self):
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            value = gf.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            destination = gf.tensor([1, 3], dtype=gf.int64)
            inverse = gf.tensor([-1, 0, -1, 1], dtype=gf.int64)
            placed = value._scatter_rows(destination, inverse, 4)

            self.assertEqual(
                placed.tolist(),
                [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
            )
            self.assertIn('"gf_tensor.scatter_rows"', placed.mlir())
            execution = placed.execution or {}
            self.assertEqual(execution.get("backend"), "cpu-llvm-jit")
            self.assertIn("scf.if", execution.get("artifacts", {}).get(
                "cpu_loop", ""))

            cotangent = gf.tensor([
                [10.0, 20.0], [1.0, 2.0],
                [30.0, 40.0], [3.0, 4.0],
            ])
            gradient = gf.autograd.grad(
                placed, value, grad_output=cotangent)
            self.assertEqual(gradient.tolist(), [[1.0, 2.0], [3.0, 4.0]])
            self.assertIn('"gf_tensor.gather"', gradient.mlir())

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_scatter_rows_forward_vjp_uses_generated_ttir(self):
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            value = gf.tensor(
                [[1.0, 2.0], [3.0, 4.0]], device="cuda:0",
                requires_grad=True,
            )
            destination = gf.tensor(
                [1, 3], dtype=gf.int64, device="cuda:0")
            inverse = gf.tensor(
                [-1, 0, -1, 1], dtype=gf.int64, device="cuda:0")
            placed = value._scatter_rows(destination, inverse, 4)

            self.assertEqual(
                placed.tolist(),
                [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 4.0]],
            )
            self.assertEqual(placed.execution["backend"], "cuda-ttir-triton")
            self.assertIn("gf_tensor_pointwise", placed.generated_code("ttir"))
            self.assertIn("tt.load", placed.generated_code("ttir"))

            cotangent = gf.tensor([
                [10.0, 20.0], [1.0, 2.0],
                [30.0, 40.0], [3.0, 4.0],
            ], device="cuda:0")
            gradient = gf.autograd.grad(
                placed, value, grad_output=cotangent)
            self.assertEqual(gradient.tolist(), [[1.0, 2.0], [3.0, 4.0]])
            self.assertEqual(gradient.execution["backend"], "cuda-ttir-triton")

    def test_native_execution_cuts_realized_operand_without_cutting_autograd(self):
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            value = gf.tensor([2.0] * 4096, requires_grad=True)
            materialized = (value * value).realize()
            output = materialized + 1.0
            self.assertEqual(output.tolist()[:2], [5.0, 5.0])

            # The second executable binds materialized storage directly and
            # therefore contains no multiply, while the semantic Python graph
            # still carries the multiply for reverse-mode differentiation.
            physical_ir = (output.execution or {}).get("ir", "")
            self.assertNotIn('"gf_tensor.mul"', physical_ir)
            self.assertIn('"gf_tensor.mul"', output.mlir())
            gradient = gf.autograd.grad(output.sum(), value)
            self.assertEqual(gradient.tolist()[:2], [4.0, 4.0])

    def test_cuda_driver_launcher_binds_provider_scratch_buffers(self):
        from tiga.codegen.ttir import _CUDADriverLauncher

        launcher = object.__new__(_CUDADriverLauncher)
        global_scratch = object()
        profile_scratch = object()
        launcher._global_scratch = global_scratch
        launcher._profile_scratch = profile_scratch
        bound = launcher._bind_arguments((1.5, True, 7, "buffer"))
        self.assertIsInstance(bound[0], ctypes.c_float)
        self.assertIsInstance(bound[1], ctypes.c_int32)
        self.assertIsInstance(bound[2], ctypes.c_int32)
        self.assertEqual(bound[3], "buffer")
        self.assertIs(bound[4], global_scratch)
        self.assertIs(bound[5], profile_scratch)

        launcher._global_scratch = None
        launcher._profile_scratch = None
        null_bound = launcher._bind_arguments(())
        self.assertEqual(tuple(value.value for value in null_bound), (0, 0))

    def test_native_radius_message_passing_and_geometry_autograd(self):
        class DistanceSum(gf.MessagePassing):
            reducer = gf.sum()

            def edge(self, src, dst, edge):
                return edge.distance * src.x

        positions = gf.tensor(
            [[0.0, 0.0], [0.3, 0.0], [0.8, 0.0]], requires_grad=True)
        source = gf.tensor([2.0, 3.0, 5.0], requires_grad=True)
        graph = gf.Graph.radius(positions, cutoff=0.6)
        kernel = DistanceSum()
        output = kernel(
            graph=graph, src={"x": source}, dst={"x": source})
        self.assertEqual(graph.resolve_csr()[0].tolist(), [0, 1, 3, 4])
        for actual, expected in zip(output.tolist(), [0.9, 3.1, 1.5]):
            self.assertAlmostEqual(actual, expected, places=5)

        position_grad, source_grad = gf.autograd.grad(
            output, (positions, source),
            grad_output=gf.tensor([1.0, 2.0, 3.0]),
            checkpoint="recompute",
        )
        expected_position = [[-7.0, 0.0], [-12.0, 0.0], [19.0, 0.0]]
        for actual_row, expected_row in zip(
            position_grad.tolist(), expected_position
        ):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(source_grad.tolist(), [0.6, 1.8, 1.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        self.assertIn('"gf_tensor.sqrt"', output.mlir(verify=True))
        self.assertIn('"gf_tensor.div"', gf.autograd.grad_mlir(
            output, positions, grad_output=gf.tensor([1.0, 2.0, 3.0])))
        self.assertEqual(
            kernel.last_variant.passes[0], "materialize-fixed-radius-snapshot")

    def test_native_periodic_radius_keeps_minimum_image_differentiable(self):
        class DistanceSum(gf.MessagePassing):
            reducer = gf.sum()

            def edge(self, src, dst, edge):
                return edge.distance * src.x

        positions = gf.tensor(
            [[0.05, 0.05], [0.95, 0.05]], requires_grad=True)
        source = gf.tensor([2.0, 3.0], requires_grad=True)
        graph = gf.Graph.radius(
            positions, cutoff=0.2, periodic=gf.tensor([1.0, 1.0]))
        output = DistanceSum()(
            graph=graph, src={"x": source}, dst={"x": source})
        for actual, expected in zip(output.tolist(), [0.3, 0.2]):
            self.assertAlmostEqual(actual, expected, places=5)
        position_grad, source_grad = gf.autograd.grad(
            output, (positions, source),
            grad_output=gf.tensor([1.0, 2.0]), checkpoint="recompute")
        for actual_row, expected_row in zip(
            position_grad.tolist(), [[7.0, 0.0], [-7.0, 0.0]]
        ):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(source_grad.tolist(), [0.2, 0.1]):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_native_radius_metric_and_select_udfs_are_batched_tensor_programs(self):
        class DistanceSum(gf.MessagePassing):
            reducer = gf.sum()

            def edge(self, src, dst, edge):
                return edge.distance * src.x

        metric_scale = gf.tensor([2.0, 1.0])

        def anisotropic_metric(src, dst, edge):
            del src, dst
            scaled = edge.displacement * metric_scale
            return (scaled * scaled).sum(axis=-1).sqrt()

        def unlike_species(src, dst, edge):
            del edge
            return src.species != dst.species

        positions = gf.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.0, 0.6]], requires_grad=True)
        source = gf.tensor([2.0, 3.0, 5.0], requires_grad=True)
        graph = gf.Graph.radius(
            positions,
            cutoff=0.9,
            fields={"species": gf.tensor([0, 1, 0], dtype=gf.int64)},
            metric=anisotropic_metric,
            select=unlike_species,
        )
        output = DistanceSum()(
            graph=graph, src={"x": source}, dst={"x": source})
        self.assertEqual(graph.resolve_csr()[0].tolist(), [0, 1, 2, 2])
        for actual, expected in zip(output.tolist(), [2.4, 1.6, 0.0]):
            self.assertAlmostEqual(actual, expected, places=5)
        position_grad, source_grad = gf.autograd.grad(
            output, (positions, source),
            grad_output=gf.tensor([1.0, 1.0, 1.0]),
            checkpoint="recompute",
        )
        expected_position = [[-10.0, 0.0], [10.0, 0.0], [0.0, 0.0]]
        for actual_row, expected_row in zip(
            position_grad.tolist(), expected_position
        ):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=5)
        for actual, expected in zip(source_grad.tolist(), [0.8, 0.8, 0.0]):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_native_coo_import_stably_canonicalizes_without_torch(self):
        source = gf.tensor([3, 1, 2, 0, 1], dtype=gf.int64)
        destination = gf.tensor([2, 0, 2, 1, 0], dtype=gf.int64)
        graph = gf.Graph.from_coo(
            source, destination, num_src=4, num_dst=4)
        row_ptr, col_idx = graph.resolve_csr()
        self.assertEqual(row_ptr.tolist(), [0, 2, 3, 5, 5])
        # Stable destination sorting preserves COO order within each row.
        self.assertEqual(col_idx.tolist(), [1, 1, 0, 3, 2])
        self.assertEqual(graph.schema.realization, "materialized_csr")
        with self.assertRaisesRegex(ValueError, "out-of-range source"):
            gf.Graph.from_coo(
                gf.tensor([-1], dtype=gf.int64),
                gf.tensor([0], dtype=gf.int64),
                num_src=1,
                num_dst=1,
            )

    def test_runtime_owned_cuda_driver_allocator_stream_event_and_kernel(self):
        runtime = gf.runtime
        try:
            buffer = runtime.Buffer(1000 * 4, device="cuda")
        except RuntimeError as error:
            if "unsupported" in str(error):
                self.skipTest("CUDA Driver runtime is unavailable")
            raise
        ptx = r"""
.version 7.0
.target sm_80
.address_size 64
.visible .entry fill(.param .u64 data, .param .u32 count) {
  .reg .pred %p;
  .reg .b32 %r<5>;
  .reg .b64 %rd<3>;
  ld.param.u64 %rd1, [data];
  ld.param.u32 %r1, [count];
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r2, %r2, %r3, %r4;
  setp.ge.u32 %p, %r2, %r1;
  @%p bra DONE;
  mul.wide.u32 %rd2, %r2, 4;
  add.s64 %rd2, %rd1, %rd2;
  st.global.u32 [%rd2], %r2;
DONE:
  ret;
}
"""
        stream = runtime.Stream("cuda")
        module = runtime.Module(ptx, device="cuda")
        kernel = module.kernel("fill")
        module.close()  # kernel owns a native module reference.
        event = kernel.launch(
            stream,
            grid=((1000 + 127) // 128,),
            block=(128,),
            arguments=(buffer, ctypes.c_uint32(1000)),
        )
        second = runtime.Stream("cuda")
        second.wait_event(event)
        completion = second.record_event()
        completion.wait()
        self.assertTrue(event.ready)
        self.assertEqual(
            struct.unpack("1000I", buffer.read()), tuple(range(1000))
        )
        prepared = kernel.prepare_async(
            lambda: stream,
            grid=((1000 + 127) // 128,), block=(128,),
            arguments=(buffer, ctypes.c_uint32(1000)),
        )
        self.assertIsNone(prepared())
        stream.record_event().wait()
        self.assertEqual(
            struct.unpack("1000I", buffer.read()), tuple(range(1000))
        )
        with self.assertRaisesRegex(RuntimeError, "host address"):
            _ = buffer.host_address

    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_cuda_blocking_write_precedes_nonblocking_stream_copy(self):
        byte_count = (1 << 20) + 17
        payload = bytes(index % 251 for index in range(byte_count))
        source = gf.runtime.Buffer(byte_count, device="cuda:0")
        destination = gf.runtime.Buffer(byte_count, device="cuda:0")
        stream = gf.runtime.Stream("cuda:0")
        completion = None
        try:
            source.write(payload)
            destination.write(bytes(byte_count))
            completion = source.copy_to(
                destination, stream=stream, bytes=byte_count)
            completion.wait()
            self.assertEqual(destination.read(), payload)
        finally:
            if completion is not None:
                completion.close()
            stream.close()
            source.close()
            destination.close()

    def test_compiler_bundle_plan_binds_without_torch_or_schedule_guessing(self):
        runtime = gf.runtime
        serialized = """{
          "schema": "tiga.executable-bundle-plan.v1",
          "resources": [{"name":"scratch","memory_space":"device",
            "layout":"packed","device":"provider:0","capacity_bytes":64,
            "snapshot_version":5,"external":false}],
          "invocations": [
            {"name":"build","task_kind":"degree-worklist",
             "phase":"materialize",
             "executable_symbol":"build_rows","depends_on":[],
             "arguments":[
               {"parameter":"row_ptr","resource":"row_ptr"},
               {"parameter":"row_ids","resource":"row_ids"}],
             "snapshot_version":5,"upper_bounds":[8,16],
             "accesses":[
               {"binding":"row_ptr","mode":"read","snapshot_version":5},
               {"binding":"row_ids","mode":"write","snapshot_version":5}]},
            {"name":"bucket0","task_kind":"degree-bucket",
             "phase":"execute",
             "executable_symbol":"compute","depends_on":["build"],
             "arguments":[
               {"parameter":"row_ids","resource":"row_ids"},
               {"parameter":"output","resource":"output"}],
             "snapshot_version":5,"bucket_ordinal":0,
             "accesses":[
               {"binding":"row_ids","mode":"read","snapshot_version":5},
               {"binding":"output","mode":"write","snapshot_version":5,
                "partitioning":"degree-worklist","partition":"bucket:0"}]},
            {"name":"bucket1","task_kind":"degree-bucket",
             "phase":"execute",
             "executable_symbol":"compute","depends_on":["build"],
             "arguments":[
               {"parameter":"row_ids","resource":"row_ids"},
               {"parameter":"output","resource":"output"}],
             "snapshot_version":5,"bucket_ordinal":1,
             "accesses":[
               {"binding":"row_ids","mode":"read","snapshot_version":5},
               {"binding":"output","mode":"write","snapshot_version":5,
                "partitioning":"degree-worklist","partition":"bucket:1"}]},
            {"name":"done","task_kind":"join",
             "phase":"execute",
             "executable_symbol":"__gf_join",
             "arguments":[],
             "depends_on":["bucket0","bucket1"],"snapshot_version":5,
             "accesses":[]}
          ],
          "terminals":["done"]
        }"""
        plan = runtime.ExecutableBundlePlan.parse(serialized)
        allocated = plan.allocate(lambda requirement: bytearray(
            requirement.capacity_bytes
        ))
        self.assertEqual(len(allocated["scratch"]), 64)
        resolved = []

        def resolver(invocation):
            resolved.append((invocation.executable_symbol, dict(invocation.metadata)))
            return lambda **resources: None

        bundle = plan.bind(resolver)
        self.assertEqual(
            bundle.execution_order, ("build", "bucket0", "bucket1", "done")
        )
        self.assertEqual([symbol for symbol, _ in resolved],
                         ["build_rows", "compute", "compute"])
        self.assertEqual(resolved[0][1]["upper_bounds"], [8, 16])
        self.assertEqual(resolved[1][1]["bucket_ordinal"], 0)
        execute = plan.bind_phase("execute", resolver)
        self.assertEqual(execute.execution_order, ("bucket0", "bucket1", "done"))
        resources = {"row_ptr": [], "row_ids": [], "output": []}
        self.assertTrue(
            bundle.submit(runtime.SynchronousProvider(), resources).ready
        )

    def test_compiler_bundle_plan_rejects_terminal_or_version_forgery(self):
        runtime = gf.runtime
        base = {
            "schema": "tiga.executable-bundle-plan.v1",
            "invocations": [{
                "name": "x", "task_kind": "local",
                "phase": "execute",
                "executable_symbol": "kernel", "depends_on": [],
                "arguments": [{"parameter": "x", "resource": "x"}],
                "snapshot_version": 1,
                "accesses": [{"binding": "x", "mode": "read",
                               "snapshot_version": 2}],
            }],
            "terminals": ["x"],
        }
        import json
        with self.assertRaisesRegex(ValueError, "mixes snapshot versions"):
            runtime.ExecutableBundlePlan.parse(json.dumps(base))
        base["invocations"][0]["accesses"][0]["snapshot_version"] = 1
        base["terminals"] = ["forged"]
        with self.assertRaisesRegex(ValueError, "terminals"):
            runtime.ExecutableBundlePlan.parse(json.dumps(base))

    def test_runtime_executable_bundle_validates_and_submits_dag(self):
        trace = []

        def build(*, source, temporary):
            temporary.extend(value * 2 for value in source)
            trace.append("build")

        def consume(*, temporary, output):
            output.extend(value + 1 for value in temporary)
            trace.append("consume")

        runtime = gf.runtime
        bundle = runtime.ExecutableBundle(
            (
                runtime.KernelInvocation(
                    "build", build,
                    arguments=(
                        runtime.ArgumentBinding("source", "source"),
                        runtime.ArgumentBinding("temporary", "temporary"),
                    ),
                    accesses=(
                        runtime.ResourceAccess("source", "read"),
                        runtime.ResourceAccess("temporary", "write"),
                    ),
                ),
                runtime.KernelInvocation(
                    "consume", consume,
                    arguments=(
                        runtime.ArgumentBinding("temporary", "temporary"),
                        runtime.ArgumentBinding("output", "output"),
                    ),
                    accesses=(
                        runtime.ResourceAccess("temporary", "read"),
                        runtime.ResourceAccess("output", "write"),
                    ),
                    depends_on=("build",),
                ),
            ),
            required_bindings=("source", "temporary", "output"),
        )
        temporary, output = [], []
        submission = bundle.submit(
            runtime.SynchronousProvider(),
            {"source": [1, 2], "temporary": temporary, "output": output},
        )
        self.assertEqual(bundle.execution_order, ("build", "consume"))
        self.assertEqual(bundle.structural_hash, bundle.structural_hash)
        self.assertIn("consume (local) <- build", bundle.explain())
        self.assertEqual(trace, ["build", "consume"])
        self.assertEqual(output, [3, 5])
        self.assertTrue(submission.ready)
        submission.wait()

        prepared_trace = []
        prepared_bundle = runtime.ExecutableBundle(
            (runtime.KernelInvocation(
                "prepared", lambda *, value: prepared_trace.append(value),
                arguments=(runtime.ArgumentBinding("value", "x"),),
                accesses=(runtime.ResourceAccess("x", "read"),),
            ),),
            required_bindings=("x",),
        ).prepare(runtime.SynchronousProvider())
        prepared_bundle.submit_resources((7,)).wait()
        self.assertEqual(prepared_trace, [7])
        with self.assertRaisesRegex(ValueError, "requires 1 resources"):
            prepared_bundle.submit_resources(())

    def test_runtime_bundle_rejects_unordered_hazards_and_cycles(self):
        runtime = gf.runtime
        writer = lambda name, dependencies=(): runtime.KernelInvocation(
            name, lambda **kwargs: None,
            arguments=(runtime.ArgumentBinding("value", "shared"),),
            accesses=(runtime.ResourceAccess("shared", "write"),),
            depends_on=dependencies,
        )
        with self.assertRaisesRegex(ValueError, "unordered resource hazard"):
            runtime.ExecutableBundle(
                (writer("left"), writer("right")),
                required_bindings=("shared",),
            )
        with self.assertRaisesRegex(ValueError, "contains a cycle"):
            runtime.ExecutableBundle(
                (writer("left", ("right",)), writer("right", ("left",))),
                required_bindings=("shared",),
            )

    def test_runtime_bundle_allows_proven_disjoint_partition_writes(self):
        runtime = gf.runtime
        invocations = tuple(
            runtime.KernelInvocation(
                f"bucket_{bucket}", lambda **kwargs: None,
                arguments=(runtime.ArgumentBinding("output", "out"),),
                accesses=(runtime.ResourceAccess(
                    "out", "write", partitioning="degree-buckets-v1",
                    partition=str(bucket)),),
            )
            for bucket in (8, 32, 64)
        )
        bundle = runtime.ExecutableBundle(
            invocations, required_bindings=("out",))
        self.assertEqual(
            bundle.execution_order, ("bucket_8", "bucket_32", "bucket_64"))
        self.assertIn(
            "out:write[degree-buckets-v1/64]", bundle.explain())

    def test_runtime_bundle_rejects_binding_and_snapshot_mismatch(self):
        runtime = gf.runtime
        invocation = runtime.KernelInvocation(
            "kernel", lambda **kwargs: None,
            arguments=(runtime.ArgumentBinding("value", "x"),),
            accesses=(runtime.ResourceAccess("x", "read", 2),),
        )
        with self.assertRaisesRegex(ValueError, "accesses snapshot 2"):
            runtime.ExecutableBundle(
                (invocation,), required_bindings=("x",), snapshot_version=1)
        bundle = runtime.ExecutableBundle(
            (invocation,), required_bindings=("x",), snapshot_version=2)
        with self.assertRaisesRegex(KeyError, "missing bindings"):
            bundle.submit(runtime.SynchronousProvider(), {})

    def test_native_dense_graph_statistics_remain_implicit(self):
        graph = gf.Graph.dense(1 << 20)
        self.assertEqual(
            graph.degree_statistics(),
            (1 << 20, 1 << 20, 1 << 40),
        )
        self.assertIn((1 << 20, 1 << 20, 1 << 40), graph.planning_key())

    def test_native_cpu_buffer_stream_and_event(self):
        self.assertEqual(gf.runtime.version(), "0.1.0a1")
        buffer = gf.runtime.Buffer(257, alignment=64)
        self.assertEqual(buffer.nbytes, 257)
        self.assertNotEqual(buffer.address, 0)

        stream = gf.runtime.Stream("cpu")
        stream.synchronize()
        event = gf.runtime.Event("cpu")
        self.assertTrue(event.ready)
        event.wait()

    def test_hierarchy_runtime_capacity_version_and_nvme_spill(self):
        runtime = gf.runtime.HierarchyRuntime(
            budgets={"ram": 32, "nvme": 64}
        )
        try:
            source = runtime.allocate(
                "field", 3, tier="ram", capacity_bytes=32
            )
            source.buffer.write(bytes(range(32)))
            with self.assertRaises(MemoryError):
                runtime.allocate("too-large", 0, tier="ram", capacity_bytes=1)
            spill = runtime.allocate(
                "field", 3, tier="nvme", capacity_bytes=32
            )
            runtime.transfer(source, spill).wait()
            source.close()
            restored = runtime.allocate(
                "field", 3, tier="ram", capacity_bytes=32
            )
            runtime.transfer(spill, restored).wait()
            self.assertEqual(restored.buffer.read(), bytes(range(32)))
            self.assertEqual(runtime.latest("field", tier="ram").version, 3)
            self.assertEqual(runtime.peak_live_bytes[gf.runtime.MemoryTier.RAM], 32)
            wrong = runtime.allocate("field", 4, tier="nvme", capacity_bytes=32)
            with self.assertRaisesRegex(ValueError, "preserve"):
                runtime.transfer(restored, wrong)
        finally:
            runtime.close()

    def test_bundle_transfer_consumes_hierarchy_instances(self):
        plan = gf.runtime.ExecutableBundlePlan.parse(r'''{
          "schema":"tiga.executable-bundle-plan.v1",
          "resources":[],
          "invocations":[{
            "name":"prefetch","task_kind":"storage-transfer",
            "phase":"materialize","executable_symbol":"__gf_transfer",
            "depends_on":[],"snapshot_version":9,
            "arguments":[{"parameter":"source","resource":"ram"},
                         {"parameter":"destination","resource":"device"}],
            "accesses":[{"binding":"ram","mode":"read","snapshot_version":9},
                        {"binding":"device","mode":"write","snapshot_version":9}]
          }],"terminals":["prefetch"]
        }''')
        hierarchy = gf.runtime.HierarchyRuntime()
        try:
            source = hierarchy.allocate(
                "u", 9, tier="ram", capacity_bytes=128
            )
            destination = hierarchy.allocate(
                "u", 9, tier="nvme", capacity_bytes=128
            )
            source.buffer.write(bytes(range(128)))
            bundle = plan.bind(lambda _invocation: self.fail("resolver called"))
            bundle.submit(
                gf.runtime.SynchronousProvider(),
                {"ram": source, "device": destination},
            ).wait()
            self.assertEqual(destination.path.read_bytes(), bytes(range(128)))
        finally:
            hierarchy.close()

    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_hierarchy_runtime_pinned_cuda_async_roundtrip(self):
        runtime = gf.runtime.HierarchyRuntime(
            budgets={"host-pinned": 4096, "device": 4096}
        )
        try:
            host = runtime.allocate(
                "tile", 0, tier="host-pinned", capacity_bytes=4096
            )
            device = runtime.allocate(
                "tile", 0, tier="device", capacity_bytes=4096,
                device="cuda:0",
            )
            restored = runtime.allocate(
                "tile", 0, tier="ram", capacity_bytes=4096
            )
            payload = bytes(index % 251 for index in range(4096))
            host.buffer.write(payload)
            runtime.transfer(host, device).wait()
            runtime.transfer(device, restored).wait()
            self.assertEqual(restored.buffer.read(), payload)
        finally:
            runtime.close()

    def test_unimplemented_device_is_explicit(self):
        with self.assertRaisesRegex(RuntimeError, "unsupported"):
            gf.runtime.Buffer(16, device="hip:0")

    def test_tensor_metadata_storage_and_reshape(self):
        value = gf.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        self.assertEqual(value.shape, (2, 2))
        self.assertEqual(value.strides, (2, 1))
        self.assertIs(value.dtype, gf.float32)
        self.assertEqual(value.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        flattened = value.reshape(4)
        self.assertEqual(flattened.tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(flattened._buffer.address, value._buffer.address)
        self.assertTrue(value.ready_event.ready)

    def test_native_cpu_real_and_complex_dtype_matrix(self):
        cases = (
            (gf.float16, 34.0),
            (gf.float32, 34.0),
            (gf.float64, 34.0),
            (gf.complex64, 34.0 + 0.0j),
            (gf.complex128, 34.0 + 0.0j),
        )
        with patch.dict(os.environ, {
            "TIGA_TENSOR_BACKEND": "native",
        }):
            for dtype, expected in cases:
                with self.subTest(dtype=dtype):
                    value = gf.tensor([1, 2, 3, 4], dtype=dtype)
                    output = (value * value + 1).sum()
                    self.assertEqual(output.tolist(), expected)
                    self.assertEqual(
                        output._execution_info["backend"], "cpu-llvm-jit"
                    )

    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_cuda_pointwise_real_dtype_matrix(self):
        import torch

        cases = (
            (torch.float16, gf.float16),
            (torch.float32, gf.float32),
            (torch.float64, gf.float64),
        )
        environment = {
            "TIGA_TENSOR_BACKEND": "native",
        }
        translator = find_gf_translate()
        if translator is None:
            self.skipTest("built gf-translate is required")
        environment["TIGA_TRANSLATE"] = translator
        with patch.dict(os.environ, environment):
            for torch_dtype, gf_dtype in cases:
                with self.subTest(dtype=gf_dtype):
                    source = torch.linspace(
                        0, 1, 8192, device="cuda", dtype=torch_dtype
                    )
                    value = gf.from_torch(source)
                    output = value * value + value
                    actual = output.to_torch()
                    torch.testing.assert_close(actual, source * source + source)
                    self.assertEqual(
                        output._execution_info["backend"], "cuda-ttir-triton"
                    )
                    self.assertIn("ptx", output._execution_info["artifacts"])

    @unittest.skipUnless(_cuda_available(), "CUDA is required")
    def test_cuda_pointwise_complex_dtype_matrix(self):
        import torch

        cases = (
            (torch.complex64, gf.complex64),
            (torch.complex128, gf.complex128),
        )
        environment = {
            "TIGA_TENSOR_BACKEND": "native",
        }
        translator = find_gf_translate()
        if translator is None:
            self.skipTest("built gf-translate is required")
        environment["TIGA_TRANSLATE"] = translator
        with patch.dict(os.environ, environment):
            for torch_dtype, gf_dtype in cases:
                with self.subTest(dtype=gf_dtype):
                    source = torch.tensor(
                        [1 + 2j, 3 - 4j] * 4096,
                        device="cuda", dtype=torch_dtype,
                    )
                    value = gf.from_torch(source)
                    constant = gf.tensor(
                        2 + 1j, dtype=gf_dtype, device="cuda"
                    )
                    output = (
                        value * value.conj() + value
                    ) / (value + constant)
                    actual = output.to_torch()
                    expected = (
                        source * source.conj() + source
                    ) / (source + torch.tensor(
                        2 + 1j, device="cuda", dtype=torch_dtype
                    ))
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual(
                        output._execution_info["backend"], "cuda-ttir-triton"
                    )
                    self.assertIn("ptx", output._execution_info["artifacts"])

    def test_reshape_inference_and_view_vjp(self):
        value = gf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        view = value.reshape(3, -1)
        self.assertEqual(view.shape, (3, 2))
        self.assertEqual(view._buffer.address, value._buffer.address)
        gradient = gf.autograd.grad((view * view).sum(), value)
        self.assertEqual(
            gradient.tolist(),
            [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]],
        )

    def test_permute_is_zero_copy_and_has_inverse_vjp(self):
        value = gf.tensor(
            [[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]],
            requires_grad=True,
        )
        view = value.permute(2, 0, 1)
        self.assertEqual(view.shape, (3, 2, 1))
        self.assertEqual(view.strides, (1, 3, 3))
        self.assertEqual(view._buffer.address, value._buffer.address)
        self.assertEqual(
            view.tolist(),
            [[[1.0], [4.0]], [[2.0], [5.0]], [[3.0], [6.0]]],
        )
        gradient = gf.autograd.grad((view * view).sum(), value)
        self.assertEqual(
            gradient.tolist(),
            [[[2.0, 4.0, 6.0]], [[8.0, 10.0, 12.0]]],
        )

    def test_transpose_and_T(self):
        value = gf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        self.assertEqual(value.transpose(0, 1).tolist(), expected)
        self.assertEqual(value.T.tolist(), expected)

    def test_noncontiguous_reshape_materializes_logical_order(self):
        value = gf.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        flattened = value.T.reshape(-1)
        self.assertIsNone(flattened._buffer)
        self.assertEqual(flattened.tolist(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        gradient = gf.autograd.grad((flattened * flattened).sum(), value)
        self.assertEqual(
            gradient.tolist(), [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]
        )

    def test_squeeze_unsqueeze_expand_and_swapaxes(self):
        value = gf.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        expanded = value.squeeze(0).unsqueeze(-1).expand(3, 4)
        self.assertEqual(expanded.shape, (3, 4))
        self.assertEqual(
            expanded.tolist(),
            [[1.0, 1.0, 1.0, 1.0],
             [2.0, 2.0, 2.0, 2.0],
             [3.0, 3.0, 3.0, 3.0]],
        )
        gradient = gf.autograd.grad(expanded.sum(), value)
        self.assertEqual(gradient.tolist(), [[4.0, 4.0, 4.0]])
        self.assertEqual(value.swapaxes(0, 1).shape, (3, 1))

    def test_numpy_style_broadcast_and_vjp(self):
        lhs = gf.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], requires_grad=True)
        rhs = gf.tensor([[[10.0], [20.0], [30.0], [40.0]]], requires_grad=True)
        output = lhs * rhs
        self.assertEqual(output.shape, (2, 4, 3))
        loss = output.sum()
        dlhs, drhs = gf.autograd.grad(loss, (lhs, rhs))
        self.assertEqual(
            dlhs.tolist(),
            [[[100.0, 100.0, 100.0]], [[100.0, 100.0, 100.0]]],
        )
        self.assertEqual(
            drhs.tolist(),
            [[[21.0], [21.0], [21.0], [21.0]]],
        )

    def test_broadcast_view_is_zero_stride_and_reduces_vjp(self):
        value = gf.tensor([[1.0], [2.0]], requires_grad=True)
        view = value.broadcast_to((3, 2, 4))
        self.assertEqual(view.strides, (0, 1, 0))
        self.assertEqual(view._buffer.address, value._buffer.address)
        gradient = gf.autograd.grad(view.sum(), value)
        self.assertEqual(gradient.tolist(), [[12.0], [12.0]])

    def test_axis_sum_keepdims_and_vjp(self):
        value = gf.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        reduced = value.sum(axis=(0, -1), keepdims=True)
        self.assertEqual(reduced.shape, (1, 2, 1))
        self.assertEqual(reduced.tolist(), [[[14.0], [22.0]]])
        gradient = gf.autograd.grad(reduced.sum(), value)
        self.assertEqual(
            gradient.tolist(),
            [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
        )

    def test_cumsum_axis_reverse_native_cpu_and_vjp(self):
        value = gf.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            prefix = value.cumsum(1)
            suffix = value.cumsum(-1, reverse=True)
            self.assertEqual(prefix.tolist(), [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]])
            self.assertEqual(suffix.tolist(), [[6.0, 5.0, 3.0], [15.0, 11.0, 6.0]])
            cotangent = gf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            gradient = gf.autograd.grad(prefix, value, grad_output=cotangent)
            self.assertEqual(
                gradient.tolist(), [[6.0, 5.0, 3.0], [15.0, 11.0, 6.0]]
            )
        self.assertIn('"gf_tensor.cumsum"', prefix.mlir())
        self.assertEqual(prefix.execution["backend"], "cpu-llvm-jit")

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_cumsum_direct_ttir_and_vjp(self):
        import torch

        torch.manual_seed(31)
        source_torch = torch.randn((7, 19, 5), device="cuda")
        cotangent_torch = torch.randn_like(source_torch)
        source = gf.from_torch(source_torch, requires_grad=True)
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            output = source.cumsum(1)
            actual = output.to_torch()
            gradient = gf.autograd.grad(
                output, source, grad_output=gf.from_torch(cotangent_torch)
            )
            actual_gradient = gradient.to_torch()
        torch.testing.assert_close(actual, torch.cumsum(source_torch, dim=1))
        expected_gradient = torch.flip(
            torch.cumsum(torch.flip(cotangent_torch, (1,)), dim=1), (1,)
        )
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(output.execution["backend"], "cuda-ttir-triton")
        self.assertIn("gf_tensor_cumsum", output.generated_code("ttir"))
        self.assertIn("scf.for", output.generated_code("ttir"))
        self.assertIn("arith.subi", gradient.generated_code("ttir"))

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_map_scan_contract_is_structurally_fused(self):
        import torch

        lanes, steps, key_width, value_width = 3, 37, 16, 13
        torch.manual_seed(41)
        q_torch = torch.randn((lanes, steps, key_width), device="cuda")
        k_torch = torch.randn_like(q_torch)
        v_torch = torch.randn((lanes, steps, value_width), device="cuda")
        q, k, v = map(gf.from_torch, (q_torch, k_torch, v_torch))
        state_shape = (lanes, steps, key_width, value_width)
        lifted = (
            k.unsqueeze(-1).broadcast_to(state_shape)
            * v.unsqueeze(-2).broadcast_to(state_shape)
        )
        output = (
            q.unsqueeze(-1).broadcast_to(state_shape) * lifted.cumsum(1)
        ).sum(axis=2)
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            actual = output.to_torch()
        expected = (
            q_torch.unsqueeze(-1)
            * (k_torch.unsqueeze(-1) * v_torch.unsqueeze(-2)).cumsum(1)
        ).sum(dim=2)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)
        ttir = output.generated_code("ttir")
        self.assertIn("gf_tensor_scan_contract", ttir)
        self.assertIn("scf.for", ttir)
        self.assertIn("tt.reduce", ttir)
        self.assertNotIn("attention", ttir.lower())

    def test_non_scalar_grad_output(self):
        value = gf.tensor([[1.0], [2.0]], requires_grad=True)
        output = value.broadcast_to((2, 3))
        cotangent = gf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        gradient = gf.autograd.grad(output, value, grad_output=cotangent)
        self.assertEqual(gradient.tolist(), [[6.0], [15.0]])

    def test_checkpoint_is_explicit_identity_with_identity_vjp(self):
        value = gf.tensor([1.0, 2.0, 3.0], requires_grad=True)
        saved = value.checkpoint()
        gradient = gf.autograd.grad(saved.sum(), value)
        self.assertEqual(saved.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(gradient.tolist(), [1.0, 1.0, 1.0])
        self.assertIn("checkpoint", saved.expression())
        self.assertIn("gf_tensor.checkpoint", saved.mlir())

    def test_autograd_checkpoint_policy_validation(self):
        value = gf.tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "checkpoint must be"):
            gf.autograd.grad(value.sum(), value, checkpoint="unknown")

    def test_native_checkpoint_planner_selects_save_or_recompute_by_budget(self):
        from tiga.compiler.native import checkpoint_plan, plan_checkpoints

        value = gf.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        index = gf.tensor([3, 1, 0], dtype=gf.int64)
        gathered = value.gather(index)
        gradient = gf.autograd.grad(
            (gathered * gathered).sum(), value, checkpoint="auto"
        )
        self.assertIn("checkpoint_candidate", gradient.expression())
        self.assertIn("gf_tensor.checkpoint_candidate", gradient.mlir())

        selected, saved_bytes, selected_ir = plan_checkpoints(gradient)
        self.assertEqual(selected, (True,))
        self.assertEqual(saved_bytes, 12)
        self.assertIn("gf_tensor.checkpoint\"", selected_ir)
        self.assertNotIn('"gf_tensor.checkpoint_candidate"', selected_ir)

        recomputed, saved_bytes, recomputed_ir = plan_checkpoints(
            gradient, memory_budget_bytes=0
        )
        self.assertEqual(recomputed, (False,))
        self.assertEqual(saved_bytes, 0)
        self.assertNotIn("gf_tensor.checkpoint\"", recomputed_ir)
        self.assertNotIn('"gf_tensor.checkpoint_candidate"', recomputed_ir)

        spilled = checkpoint_plan(
            gradient, memory_budget_bytes=0, spill_budget_bytes=12
        )
        self.assertEqual(spilled["tiers"], ("host-pinned",))
        self.assertEqual(spilled["recompute_costs"], (24,))
        self.assertEqual(spilled["spilled_bytes"], 12)
        self.assertEqual(spilled["peak_live_bytes"], 0)
        self.assertEqual(spilled["peak_spill_bytes"], 12)
        self.assertIn('storage_tier = "host-pinned"', spilled["ir"])

        from tiga.compiler.gpu_tensor import _physicalize_storage

        with patch.dict(
            os.environ, {"TIGA_CHECKPOINT_BUDGET_BYTES": "0"}
        ):
            physical, _, actual_saved, _, plan = _physicalize_storage(gradient)
        self.assertEqual(actual_saved, 0)
        self.assertEqual(plan["decisions"], (False,))
        self.assertNotIn("checkpoint_candidate", physical.expression())

    def test_shape_operation_validation(self):
        value = gf.tensor([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaisesRegex(ValueError, "at most one"):
            value.reshape(-1, -1)
        with self.assertRaisesRegex(ValueError, "permutation"):
            value.permute(0, 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            value.sum((0, 0))
        with self.assertRaisesRegex(ValueError, "broadcast"):
            value.broadcast_to((3, 2))

    def test_zero_extent_broadcast_and_reduction(self):
        lhs = gf.empty((0, 3))
        rhs = gf.tensor([[1.0, 2.0, 3.0]])
        output = lhs + rhs
        self.assertEqual(output.shape, (0, 3))
        self.assertEqual(output.tolist(), [])
        self.assertEqual(output.sum(axis=0).tolist(), [0.0, 0.0, 0.0])

    def test_native_codegen_fuses_shape_expression(self):
        lhs = gf.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
        rhs = gf.tensor([[[10.0], [20.0], [30.0], [40.0]]])
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            output = (lhs * rhs + lhs).sum((0, 2), keepdims=True)
            self.assertEqual(
                output.tolist(),
                [[[231.0], [441.0], [651.0], [861.0]]],
            )
        self.assertEqual(output.execution["backend"], "cpu-llvm-jit")
        self.assertIn("graphforge_run", output.generated_code())
        self.assertIn("llvm.func", output.generated_code())
        self.assertIn("scf.for", output.generated_code("cpu_loop"))
        self.assertIn("memref.load", output.generated_code("cpu_loop"))
        self.assertIn("gf_tensor.reduce_sum", output.execution["ir"])
        self.assertEqual(
            set(output.execution["artifacts"]),
            {"gf_tensor", "cpu_loop", "llvm"},
        )
        self.assertEqual(len(output.execution["semantic_hash"]), 64)

    def test_native_cpu_contiguous_pointwise_uses_vector_ir_and_scalar_tail(self):
        lhs = gf.tensor([float(index) for index in range(33)])
        rhs = gf.tensor([2.0] * 33)
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            output = (lhs * rhs + lhs).sqrt()
            actual = output.tolist()
        expected = [(3.0 * index) ** 0.5 for index in range(33)]
        for found, wanted in zip(actual, expected):
            self.assertAlmostEqual(found, wanted, places=5)
        loops = output.generated_code("cpu_loop")
        llvm = output.generated_code("llvm")
        self.assertIn("tiga.cpu.vector_width = 16", loops)
        self.assertIn("vector.load", loops)
        self.assertIn("vector.store", loops)
        self.assertIn("vector<16xf32>", llvm)
        # 33 elements deliberately requires the scalar [32,33) cleanup loop.
        self.assertGreaterEqual(loops.count("scf.for"), 2)
        self.assertIn("memref.store", loops)

    def test_native_matmul_ir_cpu_codegen_and_vjp(self):
        lhs = gf.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        rhs = gf.tensor(
            [[2.0, 1.0], [0.0, 3.0], [4.0, -1.0]], requires_grad=True)
        product = lhs @ rhs
        self.assertEqual(product.tolist(), [[14.0, 4.0], [32.0, 13.0]])
        self.assertIn('"gf_tensor.matmul"', product.mlir())
        cotangent = gf.tensor([[1.0, 2.0], [-1.0, 0.5]])
        dlhs, drhs = gf.autograd.grad(
            product, (lhs, rhs), grad_output=cotangent)
        self.assertEqual(
            dlhs.tolist(),
            [[4.0, 6.0, 2.0], [-1.5, 1.5, -4.5]],
        )
        self.assertEqual(
            drhs.tolist(),
            [[-3.0, 4.0], [-3.0, 6.5], [-3.0, 9.0]],
        )
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            compiled = lhs @ rhs
            self.assertEqual(compiled.tolist(), [[14.0, 4.0], [32.0, 13.0]])
        self.assertEqual(compiled.execution["backend"], "cpu-llvm-jit")
        self.assertIn("arith.mulf", compiled.generated_code("cpu_loop"))
        self.assertIn("scf.for", compiled.generated_code("cpu_loop"))

        lowered = gf.autograd.grad_mlir(
            product, lhs, grad_output=cotangent)
        self.assertNotIn('"gf_tensor.grad"', lowered)
        self.assertIn('"gf_tensor.matmul"', lowered)
        self.assertIn('"gf_tensor.permute"', lowered)

    def test_matmul_rejects_invalid_shapes_and_dtype(self):
        with self.assertRaisesRegex(ValueError, "rank-two"):
            _ = gf.tensor([1.0, 2.0]) @ gf.tensor([1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "contraction"):
            _ = gf.empty((2, 3)) @ gf.empty((4, 2))
        with self.assertRaisesRegex(TypeError, "floating-point or complex"):
            _ = gf.tensor([[1, 2]]) @ gf.tensor([[1], [2]])

    def test_tensor_mlir_is_stable_and_native_verified(self):
        def capture():
            lhs = gf.tensor([[1.0, 2.0], [3.0, 4.0]])
            rhs = gf.tensor([2.0, 3.0])
            return (lhs * rhs + lhs).sum(axis=1)

        first = capture()
        second = capture()
        self.assertEqual(first.mlir(), second.mlir())
        verified = first.mlir(verify=True)
        self.assertIn('"gf_tensor.mul"', verified)
        self.assertIn('"gf_tensor.reduce_sum"', verified)

        transposed = gf.tensor([[1.0, 2.0], [3.0, 4.0]]).T
        self.assertIn("axes = array<i64: 1, 0>", transposed.mlir())

    def test_native_mlir_vjp_pass_handles_broadcast_and_complex(self):
        value = gf.tensor([1.0 + 2.0j, 3.0 - 4.0j], requires_grad=True)
        output = value * value
        cotangent = gf.tensor([1.0 + 0.0j, 1.0 + 0.0j])
        raw = gf.autograd.grad_mlir(
            output, value, grad_output=cotangent, lower=False
        )
        self.assertIn('"gf_tensor.grad"', raw)
        lowered = gf.autograd.grad_mlir(
            output, value, grad_output=cotangent
        )
        self.assertNotIn('"gf_tensor.grad"', lowered)
        self.assertIn('"gf_tensor.conj"', lowered)
        self.assertIn('"gf_tensor.add"', lowered)

    def test_gradient_dtype_contract(self):
        with self.assertRaisesRegex(TypeError, "floating-point or complex"):
            gf.tensor([1, 2, 3], requires_grad=True)

    def test_complex_storage_views_and_inference(self):
        value = gf.tensor(
            [[1.0 + 2.0j, 3.0 - 4.0j], [-2.0 + 0.5j, 5.0 + 0.0j]],
            requires_grad=True,
        )
        self.assertIs(value.dtype, gf.complex64)
        self.assertEqual(value.nbytes, 32)
        self.assertEqual(
            value.T.tolist(),
            [[1.0 + 2.0j, -2.0 + 0.5j], [3.0 - 4.0j, 5.0 + 0.0j]],
        )
        self.assertEqual(value.conj().tolist()[0], [1.0 - 2.0j, 3.0 + 4.0j])

    def test_complex_conjugate_wirtinger_vjp(self):
        value = gf.tensor([1.0 + 2.0j, 3.0 - 4.0j], requires_grad=True)
        output = value * value
        cotangent = gf.tensor([1.0 + 0.0j, 1.0 + 0.0j])
        gradient = gf.autograd.grad(output, value, grad_output=cotangent)
        self.assertEqual(gradient.tolist(), [2.0 - 4.0j, 6.0 + 8.0j])

        conjugated = value.conj()
        conjugate_gradient = gf.autograd.grad(
            conjugated, value, grad_output=cotangent
        )
        self.assertEqual(conjugate_gradient.tolist(), [1.0 - 0.0j, 1.0 - 0.0j])

    def test_complex_output_requires_explicit_cotangent(self):
        value = gf.tensor([1.0 + 2.0j], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "explicit grad_output"):
            gf.autograd.grad(value.sum(), value)

    def test_native_complex_codegen(self):
        value = gf.tensor([1.0 + 2.0j, 3.0 - 4.0j])
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            output = (value * value.conj()).sum(
                axis=0, keepdims=True
            )
            self.assertEqual(output.tolist(), [30.0 + 0.0j])
        self.assertEqual(output.execution["backend"], "cpu-llvm-jit")
        self.assertNotIn("unrealized_conversion_cast", output.generated_code())

    def test_reverse_mode_builds_and_evaluates_symbolic_vjp(self):
        x = gf.tensor([1.0, 2.0, 3.0], requires_grad=True)
        loss = (x * x + 2.0).sum()
        gradient = gf.autograd.grad(loss, x)

        self.assertEqual(loss.tolist(), 20.0)
        self.assertEqual(gradient.tolist(), [2.0, 4.0, 6.0])
        self.assertIn("broadcast", gradient.expression())
        self.assertIn("return", gradient.expression())

    def test_joint_forward_backward_task_dag_executes(self):
        value = gf.tensor([2.0, 3.0], requires_grad=True)
        output = (value * value).sum()
        plan = gf.autograd.joint_plan(output, value, checkpoint="auto")
        actual, gradient = plan.run()
        self.assertEqual(actual.tolist(), 13.0)
        self.assertEqual(gradient.tolist(), [4.0, 6.0])
        self.assertEqual(
            plan.bundle.execution_order, ("forward", "backward:0")
        )
        self.assertIn("autograd-backward", plan.explain())

    def test_scalar_broadcast_vjp_reduces_to_scalar(self):
        x = gf.tensor([1.0, 2.0, 3.0], requires_grad=True)
        bias = gf.tensor(2.0, requires_grad=True)
        loss = (x * bias + bias).sum()
        dx, dbias = gf.autograd.grad(loss, (x, bias))
        self.assertEqual(dx.tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(dbias.tolist(), 9.0)

    def test_value_and_grad_is_functional(self):
        transformed = gf.autograd.value_and_grad(
            lambda value: (value * value).sum())
        x = gf.tensor([2.0, 4.0], requires_grad=True)
        value, gradient = transformed(x)
        self.assertEqual(value.tolist(), 20.0)
        self.assertEqual(gradient.tolist(), [4.0, 8.0])

    def test_tensor_runtime_imports_without_torch(self):
        # Exercise the package under test.  In an editable source run this is
        # ``<repo>/python``; in installed-wheel CI it is site-packages.  Pointing
        # unconditionally at the checkout would mix source Python with the
        # wheel-only native extension and would not test either installation.
        package_parent = Path(gf.__file__).resolve().parent.parent
        environment = {
            "PYTHONPATH": str(package_parent),
            "TIGA_RUNTIME_LIBRARY": str(gf.runtime._library()._name),
            "TIGA_TENSOR_BACKEND": "native",
        }
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import tiga as gf; "
                "x=gf.tensor([2.,3.], requires_grad=True); "
                "assert gf.autograd.grad((x*x).sum(), x).tolist()==[4.,6.]",
            ],
            env={**os.environ, **environment},
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_native_cuda_tensor_compiles_and_runs_without_importing_torch(self):
        repository = Path(__file__).resolve().parents[2]
        script = r'''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch import forbidden", name="torch")
        return None
sys.meta_path.insert(0, NoTorch())
import tiga as gf
view = gf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="cuda")
assert view.permute(1, 0).tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
x = gf.tensor([float(i % 17) for i in range(8192)], device="cuda")
y = x * x + x
assert y.tolist() == [float((i % 17) ** 2 + i % 17) for i in range(8192)]
assert y.execution["backend"] == "cuda-ttir-triton"
assert "ptx" in y.execution["artifacts"]
lhs = gf.tensor([[float(row + column) for column in range(64)]
                 for row in range(64)], dtype=gf.float16, device="cuda")
identity = gf.tensor([[1.0 if row == column else 0.0 for column in range(64)]
                      for row in range(64)], dtype=gf.float16, device="cuda")
product = lhs @ identity
assert product.tolist() == lhs.tolist()
assert product.execution["backend"] == "cuda-cublas-library-dispatch"
assert "cublasLtMatmul" in product.generated_code("library_dispatch")
assert "torch" not in sys.modules
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "PYTHONPATH": str(repository / "python"),
                "TIGA_RUNTIME_LIBRARY": str(gf.runtime._library()._name),
                "TIGA_TENSOR_BACKEND": "native",
                "TIGA_TRANSLATE": find_gf_translate() or "",
                "TIGA_COMPILE_WORKER": "0",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_tensor_mlir_has_no_external_tool_frontend(self):
        import tiga.compiler.tensor_mlir as tensor_mlir_module

        self.assertFalse(hasattr(tensor_mlir_module, "find_gf_opt"))
        value = gf.tensor([1.0, 2.0]) + 1.0
        self.assertIn('"gf_tensor.add"', tensor_mlir_module.tensor_mlir(value))

    def test_reducer_semantics_import_without_torch(self):
        repository = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import tiga as gf; "
                "assert gf.sum().specialization_key()==('sum',0,False); "
                "assert gf.online_softmax().name=='online_softmax'",
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(repository / "python"),
                "TIGA_RUNTIME_LIBRARY": str(gf.runtime._library()._name),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_explicit_native_storage_copy_to_torch(self):
        import torch

        value = gf.tensor([1.25, -2.5, 7.0], dtype=gf.float32)
        with self.assertRaisesRegex(RuntimeError, "zero-copy"):
            value.to_torch()
        copied = value.to_torch(copy=True)
        torch.testing.assert_close(
            copied, torch.tensor([1.25, -2.5, 7.0], dtype=torch.float32))
        self.assertNotEqual(copied.data_ptr(), value._buffer.address)

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_tensor_ttir_codegen_and_torch_zero_copy(self):
        import torch

        torch.manual_seed(11)
        torch_x = torch.randn((64, 128), device="cuda")
        torch_scale = torch.randn((128,), device="cuda")
        x = gf.from_torch(torch_x)
        scale = gf.from_torch(torch_scale)
        self.assertEqual(x.to_torch().data_ptr(), torch_x.data_ptr())
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            output = (x * scale + x).sum(axis=1)
            actual = output.to_torch()
        expected = (torch_x * torch_scale + torch_x).sum(dim=1)
        self.assertLess((actual - expected).abs().max().item(), 2e-5)
        self.assertEqual(output.execution["backend"], "cuda-ttir-triton")
        self.assertIn("ttir", output.execution["artifacts"])
        self.assertIn("ptx", output.execution["artifacts"])
        self.assertIn("gf_tensor_fused_reduce", output.generated_code("ptx"))

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_matmul_lowers_to_tensor_core_ttir_with_tail_masks(self):
        import torch

        torch.manual_seed(29)
        lhs_torch = torch.randn((65, 70), device="cuda", dtype=torch.float16)
        rhs_torch = torch.randn((70, 37), device="cuda", dtype=torch.float16)
        lhs = gf.from_torch(lhs_torch, requires_grad=True)
        rhs = gf.from_torch(rhs_torch, requires_grad=True)
        with patch.dict(os.environ, {
            "TIGA_TENSOR_BACKEND": "native",
            "TIGA_MATMUL_PROVIDER": "ttir",
        }):
            output = lhs @ rhs
            actual = output.to_torch()
        torch.testing.assert_close(
            actual, lhs_torch @ rhs_torch, rtol=2e-3, atol=2e-2)
        self.assertEqual(output.execution["backend"], "cuda-ttir-triton")
        self.assertIn("gf_tensor_matmul", output.generated_code("ttir"))
        self.assertIn("tt.dot", output.generated_code("ttir"))

        cotangent_torch = torch.randn_like(actual)
        dlhs, drhs = gf.autograd.grad(
            output, (lhs, rhs), grad_output=gf.from_torch(cotangent_torch))
        with patch.dict(os.environ, {
            "TIGA_TENSOR_BACKEND": "native",
            "TIGA_MATMUL_PROVIDER": "ttir",
        }):
            dlhs_torch = dlhs.to_torch()
            drhs_torch = drhs.to_torch()
        torch.testing.assert_close(
            dlhs_torch, cotangent_torch @ rhs_torch.T,
            rtol=2e-3, atol=3e-2)
        torch.testing.assert_close(
            drhs_torch, lhs_torch.T @ cotangent_torch,
            rtol=2e-3, atol=3e-2)
        self.assertIn("tt.dot", dlhs.generated_code("ttir"))
        self.assertIn("tt.dot", drhs.generated_code("ttir"))

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_matmul_auto_selects_runtime_library_without_losing_ir(self):
        import torch

        lhs_torch = torch.randn((129, 96), device="cuda", dtype=torch.float16)
        rhs_torch = torch.randn((96, 71), device="cuda", dtype=torch.float16)
        output = gf.from_torch(lhs_torch) @ gf.from_torch(rhs_torch)
        with patch.dict(os.environ, {
            "TIGA_TENSOR_BACKEND": "native",
            "TIGA_MATMUL_PROVIDER": "auto",
        }):
            actual = output.to_torch()
        torch.testing.assert_close(
            actual, lhs_torch @ rhs_torch, rtol=2e-3, atol=2e-2)
        self.assertEqual(
            output.execution["backend"], "cuda-cublas-library-dispatch")
        self.assertIn('"gf_tensor.matmul"', output.generated_code("gf_tensor"))
        self.assertIn("cublasLtMatmul", output.generated_code("library_dispatch"))

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_relation_edge_vjp_auto_checkpoint_codegen(self):
        import torch

        class Weighted(gf.MessagePassing):
            reducer = gf.sum()

            def edge(self, src, dst, edge):
                del dst
                return edge.weight * src.x

        nodes, degree = 1024, 8
        edges = nodes * degree
        torch.manual_seed(17)
        row_ptr = torch.arange(
            0, edges + 1, degree, dtype=torch.int64, device="cuda")
        col_idx = torch.randint(
            nodes, (edges,), dtype=torch.int64, device="cuda")
        torch_x = torch.randn(nodes, device="cuda")
        torch_weight = torch.randn(edges, device="cuda")
        torch_dy = torch.randn(nodes, device="cuda")
        graph = gf.Graph.from_csr(
            gf.from_torch(row_ptr), gf.from_torch(col_idx), num_src=nodes)
        x = gf.from_torch(torch_x, requires_grad=True)
        weight = gf.from_torch(torch_weight, requires_grad=True)
        output = Weighted()(
            graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
        gradient = gf.autograd.grad(
            output, weight, grad_output=gf.from_torch(torch_dy),
            checkpoint="auto")

        self.assertIn("checkpoint_candidate", gradient.expression())
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            actual = gradient.to_torch()
        destination = torch.arange(nodes, device="cuda").repeat_interleave(degree)
        expected = torch_dy[destination] * torch_x[col_idx]
        torch.testing.assert_close(actual, expected)
        self.assertEqual(gradient.execution["backend"], "cuda-ttir-triton")
        self.assertEqual(gradient.execution["saved_bytes"], edges * 4)
        self.assertEqual(
            gradient.execution["checkpoint_plan"]["decisions"], (True,)
        )
        self.assertIn("gf_tensor_pointwise", gradient.generated_code("ptx"))

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_checkpoint_host_spill_plan_is_physically_consumed(self):
        import torch

        class Weighted(gf.MessagePassing):
            reducer = gf.sum()

            def edge(self, src, dst, edge):
                del dst
                return edge.weight * src.x

        nodes, degree = 256, 4
        edges = nodes * degree
        values = torch.randn(nodes, device="cuda")
        index = torch.randint(nodes, (edges,), device="cuda", dtype=torch.int64)
        row_ptr = torch.arange(
            0, edges + 1, degree, device="cuda", dtype=torch.int64
        )
        weight = torch.randn(edges, device="cuda")
        cotangent = torch.randn(nodes, device="cuda")
        graph = gf.Graph.from_csr(
            gf.from_torch(row_ptr), gf.from_torch(index), num_src=nodes
        )
        source = gf.from_torch(values, requires_grad=True)
        edge_weight = gf.from_torch(weight, requires_grad=True)
        output = Weighted()(
            graph=graph,
            src={"x": source},
            dst={},
            edge={"weight": edge_weight},
        )
        gradient = gf.autograd.grad(
            output,
            edge_weight,
            grad_output=gf.from_torch(cotangent),
            checkpoint="auto",
        )
        environment = {
            "TIGA_TENSOR_BACKEND": "native",
            "TIGA_CHECKPOINT_BUDGET_BYTES": "0",
            "TIGA_CHECKPOINT_SPILL_BUDGET_BYTES": str(edges * 4),
        }
        with patch.dict(os.environ, environment):
            actual = gradient.to_torch()
        destination = torch.arange(nodes, device="cuda").repeat_interleave(degree)
        expected = cotangent[destination] * values[index]
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        plan = gradient.execution["checkpoint_plan"]
        self.assertEqual(plan["tiers"], ("host-pinned",))
        self.assertEqual(plan["planned_spilled_bytes"], edges * 4)
        self.assertEqual(plan["actual_spilled_bytes"], edges * 4)
        self.assertEqual(plan["peak_live_bytes"], 0)
        self.assertEqual(plan["peak_spill_bytes"], edges * 4)
        self.assertGreater(plan["spill_transfer_ms"], 0.0)

    @unittest.skipUnless(_cuda_available(), "CUDA is unavailable")
    def test_cuda_vector_relation_edge_vjp_fuses_gather_dot(self):
        import torch

        class WeightedFeatures(gf.MessagePassing):
            reducer = gf.sum()

            def edge(self, src, dst, edge):
                del dst
                return edge.weight * src.x

        nodes, degree, features = 512, 8, 16
        edges = nodes * degree
        torch.manual_seed(23)
        row_ptr = torch.arange(
            0, edges + 1, degree, dtype=torch.int64, device="cuda")
        col_idx = torch.randint(
            nodes, (edges,), dtype=torch.int64, device="cuda")
        destination = torch.arange(
            nodes, device="cuda").repeat_interleave(degree)
        torch_x = torch.randn(nodes, features, device="cuda")
        torch_weight = torch.randn(edges, 1, device="cuda")
        torch_dy = torch.randn(nodes, features, device="cuda")
        graph = gf.Graph.from_csr(
            gf.from_torch(row_ptr), gf.from_torch(col_idx), num_src=nodes)
        x = gf.from_torch(torch_x, requires_grad=True)
        weight = gf.from_torch(torch_weight, requires_grad=True)
        output = WeightedFeatures()(
            graph=graph, src={"x": x}, dst={}, edge={"weight": weight})
        gradient = gf.autograd.grad(
            output, weight, grad_output=gf.from_torch(torch_dy),
            checkpoint="auto")

        self.assertEqual(gradient.shape, (edges, 1))
        self.assertIn("checkpoint_candidate", gradient.expression())
        self.assertIn("sum", gradient.expression())
        with patch.dict(os.environ, {"TIGA_TENSOR_BACKEND": "native"}):
            actual = gradient.to_torch()
        expected = (
            torch_dy[destination] * torch_x[col_idx]
        ).sum(dim=1, keepdim=True)
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        self.assertEqual(gradient.execution["backend"], "cuda-ttir-triton")
        self.assertEqual(
            gradient.execution["saved_bytes"], edges * features * 4)
        self.assertEqual(
            gradient.execution["checkpoint_plan"]["decisions"], (True,)
        )
        self.assertIn("gf_tensor_fused_reduce", gradient.generated_code("ptx"))


if __name__ == "__main__":
    unittest.main()
