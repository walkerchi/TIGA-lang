from __future__ import annotations

import multiprocessing
import os
import json
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import graphforge as gf
from graphforge.distributed import (
    DeviceBufferSlice, DistributedRuntime, DistributedTaskResolver,
    MPITransportProvider, NCCLTransportProvider, PipeTransport, ShardedField,
    TransportCapabilities, create_transport, nccl_unique_id, owned_range,
    register_transport, transport_conformance,
)


class ShardedNeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


def _halo_worker(rank, endpoint, halo, queue):
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    owned = b"".join(
        struct.pack("q", entity * 10)
        for entity in range(halo.owned_begin, halo.owned_end)
    )
    with DistributedRuntime(transport) as runtime:
        completion = runtime.exchange_halo(halo, owned, element_bytes=8)
        self_consistent_ready = isinstance(completion.ready, bool)
        completion.wait()
        ghosts = completion.result()
    queue.put((rank, {
        entity: struct.unpack("q", ghosts.value_bytes(entity))[0]
        for entity in ghosts.ids
    }, self_consistent_ready))


def _bundle_halo_worker(rank, endpoint, halo, serialized_plan, queue):
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    plan = gf.runtime.ExecutableBundlePlan.parse(serialized_plan)
    field = ShardedField(
        halo,
        b"".join(
            struct.pack("q", entity * 10)
            for entity in range(halo.owned_begin, halo.owned_end)
        ),
        8,
    )
    resolver = DistributedTaskResolver(halo, transport, element_bytes=8)
    resources = resolver.allocate_communication(plan)
    resources["field"] = field
    plan.bind(resolver).submit(gf.runtime.SynchronousProvider(), resources).wait()
    queue.put((rank, {
        entity: struct.unpack("q", field.ghosts.value_bytes(entity))[0]
        for entity in field.ghosts.ids
    }))


def _sharded_message_worker(rank, endpoint, row_ptr, col_idx, queue):
    entities = len(row_ptr) - 1
    begin, end = owned_range(entities, 2, rank)
    graph = gf.Graph.from_csr(
        gf.tensor(row_ptr, dtype=gf.int64),
        gf.tensor(col_idx, dtype=gf.int64),
        num_src=entities,
        validate="full",
    ).halo(gf.DeviceMesh("cpu", 2), depth=1)
    local_x = gf.tensor([float(entity) for entity in range(begin, end)])
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    with DistributedRuntime(transport):
        output = ShardedNeighborSum()(
            graph=graph, src={"x": local_x}, dst={})
    queue.put((rank, output.tolist()))


def _paged_sharded_message_worker(rank, endpoint, path, queue):
    graph = gf.load(path).halo(gf.DeviceMesh("cpu", 2), depth=1)
    begin, end = owned_range(graph.schema.num_dst, 2, rank)
    local_x = gf.tensor([float(entity) for entity in range(begin, end)])
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    with DistributedRuntime(transport):
        output = ShardedNeighborSum()(
            graph=graph, src={"x": local_x}, dst={})
    queue.put((rank, output.tolist(), graph.schema.realization))


def _sharded_vjp_worker(rank, endpoint, row_ptr, col_idx, queue):
    entities = len(row_ptr) - 1
    begin, end = owned_range(entities, 2, rank)
    graph = gf.Graph.from_csr(
        gf.tensor(row_ptr, dtype=gf.int64),
        gf.tensor(col_idx, dtype=gf.int64),
        num_src=entities,
        validate="full",
    ).halo(gf.DeviceMesh("cpu", 2), depth=1)
    local_x = gf.tensor(
        [float(entity) for entity in range(begin, end)], requires_grad=True)
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    with DistributedRuntime(transport):
        output = ShardedNeighborSum()(
            graph=graph, src={"x": local_x}, dst={})
        gradient = gf.autograd.grad(output.sum(), local_x)
        values = gradient.tolist()
    queue.put((rank, values, gradient.expression()))


def _sharded_gpu_message_worker(rank, endpoint, row_ptr, col_idx, queue):
    device = "cuda:0"
    entities = len(row_ptr) - 1
    begin, end = owned_range(entities, 2, rank)
    graph = gf.Graph.from_csr(
        gf.tensor(row_ptr, dtype=gf.int64, device=device),
        gf.tensor(col_idx, dtype=gf.int64, device=device),
        num_src=entities, validate="full",
    ).halo(gf.DeviceMesh("cuda", 2), depth=1)
    local_x = gf.tensor(
        [float(entity) for entity in range(begin, end)], device=device)
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    os.environ["GRAPHFORGE_TENSOR_BACKEND"] = "native"
    with DistributedRuntime(transport):
        output = ShardedNeighborSum()(
            graph=graph, src={"x": local_x}, dst={})
        values = output.tolist()
        backend = output.execution["backend"]
    queue.put((rank, values, backend))


def _sharded_gpu_vjp_worker(rank, endpoint, row_ptr, col_idx, queue):
    device = "cuda:0"
    entities = len(row_ptr) - 1
    begin, end = owned_range(entities, 2, rank)
    graph = gf.Graph.from_csr(
        gf.tensor(row_ptr, dtype=gf.int64, device=device),
        gf.tensor(col_idx, dtype=gf.int64, device=device),
        num_src=entities, validate="full",
    ).halo(gf.DeviceMesh("cuda", 2), depth=1)
    local_x = gf.tensor(
        [float(entity) for entity in range(begin, end)], device=device,
        requires_grad=True)
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    os.environ["GRAPHFORGE_TENSOR_BACKEND"] = "native"
    with DistributedRuntime(transport):
        output = ShardedNeighborSum()(
            graph=graph, src={"x": local_x}, dst={})
        gradient = gf.autograd.grad(output.sum(), local_x)
        values = gradient.tolist()
        backend = gradient.execution["backend"]
    queue.put((rank, values, backend))


class DistributedRuntimeTest(unittest.TestCase):
    _HALO_BUNDLE = r'''{
      "schema":"graphforge.executable-bundle-plan.v1",
      "resources":[
        {"name":"halo-send:0","memory_space":"remote","device":"mesh",
         "layout":"packed","capacity_bytes":64,"snapshot_version":0,"external":false},
        {"name":"halo-receive:0","memory_space":"remote","device":"mesh",
         "layout":"packed","capacity_bytes":64,"snapshot_version":0,"external":false}
      ],
      "invocations":[
        {"name":"pack","task_kind":"halo-pack","phase":"execute",
         "executable_symbol":"__gf_halo_pack","depends_on":[],"snapshot_version":0,
         "arguments":[{"parameter":"source","resource":"field"},
                      {"parameter":"send","resource":"halo-send:0"}],
         "accesses":[{"binding":"field","mode":"read","snapshot_version":0},
                     {"binding":"halo-send:0","mode":"write","snapshot_version":0}]},
        {"name":"exchange","task_kind":"halo-exchange","phase":"execute",
         "executable_symbol":"__gf_halo_exchange","depends_on":["pack"],
         "snapshot_version":0,
         "arguments":[{"parameter":"send","resource":"halo-send:0"},
                      {"parameter":"receive","resource":"halo-receive:0"}],
         "accesses":[{"binding":"halo-send:0","mode":"read","snapshot_version":0},
                     {"binding":"halo-receive:0","mode":"write","snapshot_version":0}]},
        {"name":"unpack","task_kind":"halo-unpack","phase":"execute",
         "executable_symbol":"__gf_halo_unpack","depends_on":["exchange"],
         "snapshot_version":0,
         "arguments":[{"parameter":"receive","resource":"halo-receive:0"},
                      {"parameter":"destination","resource":"field"}],
         "accesses":[{"binding":"halo-receive:0","mode":"read","snapshot_version":0},
                     {"binding":"field","mode":"write","snapshot_version":0}]}
      ],"terminals":["unpack"]
    }'''

    def test_transport_plugin_abi_binds_mpi_compatible_communicator(self):
        class LoopbackCommunicator:
            def __init__(self):
                self.messages = {}
            def Get_rank(self):
                return 0
            def Get_size(self):
                return 1
            def send(self, payload, *, dest, tag):
                self.messages[(dest, tag)] = payload
            def recv(self, *, source, tag):
                return self.messages.pop((source, tag))

        communicator = LoopbackCommunicator()
        transport = create_transport(
            "mpi", communicator=communicator, tag=17)
        transport.send(0, {"halo": (1, 2)})
        self.assertEqual(transport.receive(0), {"halo": (1, 2)})
        report = transport_conformance(MPITransportProvider())
        self.assertEqual(report["abi_version"], 1)
        self.assertEqual(report["backends"], ("mpi",))
        self.assertFalse(report["device_direct"])
        with DistributedRuntime.from_provider(
            "mpi", communicator=communicator) as runtime:
            self.assertEqual(runtime.transport.world_size, 1)

        class InvalidProvider:
            capabilities = TransportCapabilities(
                "invalid", ("test",), False, False)
            def create(self, **options):
                del options
                return object()
        register_transport("invalid-test", InvalidProvider(), replace=True)
        with self.assertRaisesRegex(TypeError, "invalid transport"):
            create_transport("invalid-test")

    def test_nccl_provider_executes_device_direct_buffer_exchange(self):
        try:
            gf.runtime.cuda_compute_capability("cuda:0")
            communicator_id = nccl_unique_id()
            transport = create_transport(
                "nccl", rank=0, world_size=1,
                communicator_id=communicator_id, device="cuda:0",
            )
        except (ModuleNotFoundError, RuntimeError) as error:
            self.skipTest(f"NCCL/CUDA runtime is unavailable: {error}")
        report = transport_conformance(NCCLTransportProvider())
        self.assertTrue(report["device_direct"])
        self.assertTrue(report["asynchronous"])
        self.assertFalse(report["object_messages"])
        self.assertEqual(report["backends"], ("nccl", "cuda"))

        byte_count = (1 << 20) + 17
        source = gf.runtime.Buffer(byte_count, device="cuda:0")
        destination = gf.runtime.Buffer(byte_count, device="cuda:0")
        external_source = gf.runtime.Buffer.wrap_address(
            source.address, source.nbytes, device="cuda:0", owner=source)
        stream = gf.runtime.Stream("cuda:0")
        payload = bytes(index % 251 for index in range(byte_count))
        source.write(payload)
        destination.write(bytes(byte_count))
        try:
            completion = transport.exchange_device(
                ((0, DeviceBufferSlice(external_source, 0, len(payload))),),
                ((0, DeviceBufferSlice(destination, 0, len(payload))),),
                stream=stream,
            )
            completion.wait()
            self.assertEqual(destination.read(), payload)
            self.assertGreater(transport.version, 0)
            with self.assertRaisesRegex(NotImplementedError, "device-direct"):
                transport.send(0, payload)
        finally:
            stream.close()
            external_source.close()
            source.close()
            destination.close()
            transport.close()

    def test_device_transport_binds_full_rank_local_message_passing(self):
        try:
            gf.runtime.cuda_compute_capability("cuda:0")
        except RuntimeError as error:
            self.skipTest(f"CUDA runtime is unavailable: {error}")

        class DeviceFixtureTransport:
            rank = 0
            world_size = 2

            def __init__(self):
                # Rank zero's ring ghosts are global entities 4 and 7, in the
                # exact receive order derived from the graph snapshot.
                self.remote = gf.runtime.Buffer(8, device="cuda:0")
                self.remote.write(struct.pack("ff", 4.0, 7.0))
                self.exchanges = []

            def send(self, peer, payload):
                del peer, payload
                raise AssertionError("host transport path was selected")

            def receive(self, peer):
                del peer
                raise AssertionError("host transport path was selected")

            def exchange_device(self, sends, receives, *, stream):
                self.exchanges.append((sends, receives))
                if len(self.exchanges) == 2:
                    # Rank one's rows contribute one additional cotangent to
                    # rank-zero-owned entities 0 and 3.
                    self.remote.write(struct.pack("ff", 1.0, 1.0))
                completions = [
                    self.remote.copy_to(
                        region.buffer, stream=stream,
                        destination_offset=region.offset,
                        bytes=region.byte_count,
                    )
                    for _peer, region in receives
                ]
                return completions[-1]

        entities = 8
        rows = [3 * row for row in range(entities + 1)]
        columns = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        graph = gf.Graph.from_csr(
            gf.tensor(rows, dtype=gf.int64, device="cuda:0"),
            gf.tensor(columns, dtype=gf.int64, device="cuda:0"),
            num_src=entities, validate="full",
        ).halo(gf.DeviceMesh("cuda", 2), depth=1)
        local_x = gf.tensor(
            [0.0, 1.0, 2.0, 3.0], device="cuda:0", requires_grad=True)
        transport = DeviceFixtureTransport()
        try:
            with mock.patch.dict(os.environ, {"GRAPHFORGE_TENSOR_BACKEND": "native"}):
                with DistributedRuntime(transport):
                    output = ShardedNeighborSum()(
                        graph=graph, src={"x": local_x}, dst={})
                    self.assertEqual(output.tolist(), [8.0, 3.0, 6.0, 9.0])
                    gradient = gf.autograd.grad(output.sum(), local_x)
                    self.assertEqual(gradient.tolist(), [3.0, 3.0, 3.0, 3.0])
            self.assertEqual(len(transport.exchanges), 2)
            self.assertTrue(all(
                len(sends) == len(receives) == 1
                for sends, receives in transport.exchanges
            ))
            self.assertIn("transport=device-direct", output.expression())
            self.assertEqual(output.execution["backend"], "cuda-ttir-triton")
            self.assertEqual(gradient.execution["backend"], "cuda-ttir-triton")
        finally:
            transport.remote.close()

    def test_real_mpi_two_rank_paged_forward_and_vjp(self):
        mpiexec = shutil.which("mpiexec")
        try:
            from mpi4py import MPI  # noqa: F401
        except (ImportError, RuntimeError):
            self.skipTest("mpi4py with an MPI runtime is unavailable")
        if mpiexec is None:
            # Wheels may install the launcher next to the active interpreter
            # without activating that directory in PATH.
            candidate = Path(os.path.dirname(os.sys.executable)) / "mpiexec"
            mpiexec = str(candidate) if candidate.exists() else None
        if mpiexec is None:
            self.skipTest("mpiexec is unavailable")
        entities = 8
        rows = [3 * row for row in range(entities + 1)]
        columns = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path = root / "ring.gfg"
            gf.save(gf.Graph.from_csr(
                gf.tensor(rows, dtype=gf.int64),
                gf.tensor(columns, dtype=gf.int64),
                num_src=entities, validate="full"), graph_path)
            environment = os.environ.copy()
            source_path = str(Path(__file__).resolve().parents[2] / "python")
            bindings = str(Path(__file__).resolve().parents[2]
                           / "build/mlir-22.1.8/python_bindings")
            environment["PYTHONPATH"] = os.pathsep.join((source_path, bindings))
            completed = subprocess.run(
                [mpiexec, "-n", "2", os.sys.executable,
                 str(Path(__file__).with_name("_mpi_halo_case.py")),
                 str(graph_path), str(root)],
                env=environment, text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(
                completed.returncode, 0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
            records = [json.loads((root / f"rank-{rank}.json").read_text())
                       for rank in range(2)]
            self.assertEqual(records[0]["output"], [8.0, 3.0, 6.0, 9.0])
            self.assertEqual(records[1]["output"], [12.0, 15.0, 18.0, 13.0])
            self.assertTrue(all(
                record["gradient"] == [3.0, 3.0, 3.0, 3.0]
                and record["realization"] == "paged_csr"
                for record in records))

    def test_real_two_process_owner_ghost_halo_exchange(self):
        # Each destination reads itself and its two ring neighbors.
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source
            for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        halos = gf.collective_halo_maps(
            row_ptr, col_idx, num_entities=entities, world_size=2
        )
        self.assertEqual(halos[0].ghost_ids, (4, 7))
        self.assertEqual(halos[1].ghost_ids, (0, 3))

        context = multiprocessing.get_context("spawn")
        first, second = context.Pipe(duplex=True)
        queue = context.Queue()
        processes = (
            context.Process(target=_halo_worker, args=(0, first, halos[0], queue)),
            context.Process(target=_halo_worker, args=(1, second, halos[1], queue)),
        )
        for process in processes:
            process.start()
        received = [queue.get(timeout=10) for _ in processes]
        results = {rank: values for rank, values, _ready in received}
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(results[0], {4: 40, 7: 70})
        self.assertEqual(results[1], {0: 0, 3: 30})
        self.assertTrue(all(ready for _rank, _values, ready in received))

    def test_persistent_graph_is_paged_and_sharded_execution_is_bounded(self):
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ring.gfg"
            gf.save(
                gf.Graph.from_csr(
                    gf.tensor(row_ptr, dtype=gf.int64),
                    gf.tensor(col_idx, dtype=gf.int64),
                    num_src=entities, validate="full"),
                path,
            )
            paged = gf.load(path)
            self.assertIs(type(paged), gf.Graph)
            self.assertEqual(paged.schema.realization, "paged_csr")
            self.assertEqual(paged.num_edges, len(col_idx))
            self.assertEqual(paged.paged_rows(4, 6), (
                (0, 3, 6), tuple(col_idx[12:18])))
            self.assertIn("backing=nvme:", paged.explain())
            copied_path = Path(directory) / "ring-copy.gfg"
            with mock.patch.object(
                paged, "resolve_csr",
                side_effect=AssertionError("paged copy materialized topology"),
            ):
                gf.save(paged, copied_path)
            self.assertEqual(
                gf.load(copied_path).paged_rows(4, 6),
                ((0, 3, 6), tuple(col_idx[12:18])))

            context = multiprocessing.get_context("spawn")
            first, second = context.Pipe(duplex=True)
            queue = context.Queue()
            processes = (
                context.Process(
                    target=_paged_sharded_message_worker,
                    args=(0, first, str(path), queue)),
                context.Process(
                    target=_paged_sharded_message_worker,
                    args=(1, second, str(path), queue)),
            )
            for process in processes:
                process.start()
            received = [queue.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                self.assertEqual(process.exitcode, 0)
            results = {rank: values for rank, values, _kind in received}
            expected = {
                rank: [
                    float((destination - 1) % entities + destination
                          + (destination + 1) % entities)
                    for destination in range(*owned_range(entities, 2, rank))
                ]
                for rank in range(2)
            }
            self.assertEqual(results, expected)
            self.assertTrue(all(kind == "paged_csr" for _r, _v, kind in received))

    def test_persistent_graph_rejects_overwrite_and_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.gfg"
            graph = gf.Graph.from_csr(
                gf.tensor([0, 1], dtype=gf.int32),
                gf.tensor([0], dtype=gf.int32),
                num_src=1, validate="full")
            gf.save(graph, path)
            with self.assertRaises(FileExistsError):
                gf.save(graph, path)
            (path / "col_idx.bin").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "col_idx size"):
                gf.load(path)

    def test_halo_transport_rejects_inconsistent_owned_buffer(self):
        halos = gf.collective_halo_maps(
            [0, 1, 2, 3, 4], [3, 0, 1, 2],
            num_entities=4, world_size=2,
        )
        class UnusedTransport:
            rank = 0
            world_size = 2
            def send(self, peer, payload):
                raise AssertionError
            def receive(self, peer):
                raise AssertionError
        with self.assertRaisesRegex(ValueError, "expected"):
            gf.exchange_halo(
                halos[0], b"short", element_bytes=8,
                transport=UnusedTransport(),
            )

    def test_compiler_bundle_binds_real_pack_exchange_unpack_executables(self):
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        halos = gf.collective_halo_maps(
            row_ptr, col_idx, num_entities=entities, world_size=2)
        context = multiprocessing.get_context("spawn")
        first, second = context.Pipe(duplex=True)
        queue = context.Queue()
        processes = (
            context.Process(
                target=_bundle_halo_worker,
                args=(0, first, halos[0], self._HALO_BUNDLE, queue)),
            context.Process(
                target=_bundle_halo_worker,
                args=(1, second, halos[1], self._HALO_BUNDLE, queue)),
        )
        for process in processes:
            process.start()
        results = dict(queue.get(timeout=10) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(results[0], {4: 40, 7: 70})
        self.assertEqual(results[1], {0: 0, 3: 30})

    def test_graph_halo_automatically_executes_rank_local_message_passing(self):
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        context = multiprocessing.get_context("spawn")
        first, second = context.Pipe(duplex=True)
        queue = context.Queue()
        processes = (
            context.Process(
                target=_sharded_message_worker,
                args=(0, first, row_ptr, col_idx, queue)),
            context.Process(
                target=_sharded_message_worker,
                args=(1, second, row_ptr, col_idx, queue)),
        )
        for process in processes:
            process.start()
        results = dict(queue.get(timeout=10) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        expected = [
            float((row - 1) % entities + row + (row + 1) % entities)
            for row in range(entities)
        ]
        self.assertEqual(results[0], expected[:4])
        self.assertEqual(results[1], expected[4:])

    def test_graph_halo_vjp_reverse_exchanges_ghost_cotangents(self):
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        context = multiprocessing.get_context("spawn")
        first, second = context.Pipe(duplex=True)
        queue = context.Queue()
        processes = (
            context.Process(
                target=_sharded_vjp_worker,
                args=(0, first, row_ptr, col_idx, queue)),
            context.Process(
                target=_sharded_vjp_worker,
                args=(1, second, row_ptr, col_idx, queue)),
        )
        for process in processes:
            process.start()
        received = [queue.get(timeout=10) for _ in processes]
        results = {rank: values for rank, values, _expr in received}
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        # Every ring entity appears in exactly three destination rows.  The
        # two cross-rank appearances reach the owner through reverse halo.
        self.assertEqual(results[0], [3.0, 3.0, 3.0, 3.0])
        self.assertEqual(results[1], [3.0, 3.0, 3.0, 3.0])
        self.assertTrue(all(
            "distributed_halo_reverse" in expression
            for _rank, _values, expression in received
        ))

    def test_gpu_rank_local_fields_stage_halo_and_run_native_ttir(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CUDA availability probe requires optional Torch")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        context = multiprocessing.get_context("spawn")
        first, second = context.Pipe(duplex=True)
        queue = context.Queue()
        processes = (
            context.Process(
                target=_sharded_gpu_message_worker,
                args=(0, first, row_ptr, col_idx, queue)),
            context.Process(
                target=_sharded_gpu_message_worker,
                args=(1, second, row_ptr, col_idx, queue)),
        )
        for process in processes:
            process.start()
        received = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)
        results = {rank: values for rank, values, _backend in received}
        expected = [
            float((row - 1) % entities + row + (row + 1) % entities)
            for row in range(entities)
        ]
        self.assertEqual(results[0], expected[:4])
        self.assertEqual(results[1], expected[4:])
        self.assertTrue(all(
            backend == "cuda-ttir-triton"
            for _rank, _values, backend in received))

    def test_gpu_reverse_halo_returns_ghost_cotangents_to_owner(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CUDA availability probe requires optional Torch")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        context = multiprocessing.get_context("spawn")
        first, second = context.Pipe(duplex=True)
        queue = context.Queue()
        processes = (
            context.Process(
                target=_sharded_gpu_vjp_worker,
                args=(0, first, row_ptr, col_idx, queue)),
            context.Process(
                target=_sharded_gpu_vjp_worker,
                args=(1, second, row_ptr, col_idx, queue)),
        )
        for process in processes:
            process.start()
        received = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(
            {rank: values for rank, values, _backend in received},
            {0: [3.0] * 4, 1: [3.0] * 4},
        )
        self.assertTrue(all(
            backend == "cuda-ttir-triton"
            for _rank, _values, backend in received))


if __name__ == "__main__":
    unittest.main()
