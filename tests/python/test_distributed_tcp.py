from __future__ import annotations

import json
import multiprocessing
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

import tiga as gf
from tiga.distributed import (
    DistributedRuntime,
    TCPTransport,
    TCPTransportProvider,
    create_transport,
    discover_transports,
    owned_range,
    transport_conformance,
)


class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


def _unused_port() -> int:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _tcp_halo_worker(rank, port, graph_path, result_dir):
    graph = gf.load(graph_path).halo(gf.DeviceMesh("cpu", 2), depth=1)
    begin, end = owned_range(graph.schema.num_dst, 2, rank)
    local_x = gf.tensor(
        [float(entity) for entity in range(begin, end)], requires_grad=True)
    if rank == 0:
        transport = create_transport(
            "tcp", rank=0, world_size=2, port=port, bind="127.0.0.1")
    else:
        transport = create_transport(
            "tcp", rank=1, world_size=2, host="127.0.0.1", port=port,
            timeout=15.0)
    try:
        with DistributedRuntime(transport):
            output = NeighborSum()(graph=graph, src={"x": local_x}, dst={})
            gradient = gf.autograd.grad(output.sum(), local_x)
            payload = {
                "output": output.tolist(),
                "gradient": gradient.tolist(),
            }
    finally:
        transport.close()
    (Path(result_dir) / f"rank-{rank}.json").write_text(
        json.dumps(payload), encoding="utf-8")


class TCPTransportTest(unittest.TestCase):
    def test_tcp_provider_is_registered_and_conformant(self):
        self.assertIn("tcp", discover_transports())
        report = transport_conformance(TCPTransportProvider())
        self.assertEqual(report["abi_version"], 1)
        self.assertEqual(report["backends"], ("tcp",))
        self.assertFalse(report["device_direct"])
        with self.assertRaisesRegex(TypeError, "missing TCP transport option"):
            create_transport("tcp", rank=0, world_size=2)

    def test_tcp_two_rank_byte_exact_messages(self):
        port = _unused_port()
        outcome = {}

        def run_host():
            outcome["host"] = TCPTransport.host(
                rank=0, world_size=2, host="127.0.0.1", port=port)

        thread = threading.Thread(target=run_host)
        thread.start()
        joined = TCPTransport.join(
            rank=1, world_size=2, host="127.0.0.1", port=port, timeout=10.0)
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        hosted = outcome["host"]
        self.assertEqual((hosted.rank, hosted.world_size), (0, 2))
        self.assertEqual(hosted.port, port)
        try:
            hosted.send(1, {"ids": (4, 7), "payload": b"abc"})
            self.assertEqual(
                joined.receive(0), {"ids": (4, 7), "payload": b"abc"})
            joined.send(0, ("reverse", 7))
            self.assertEqual(hosted.receive(1), ("reverse", 7))
            for index in range(64):
                hosted.send_bytes(1, struct.pack("=i", index))
            for index in range(64):
                self.assertEqual(
                    struct.unpack("=i", joined.receive_bytes(0, 4))[0], index)
            payload = bytes(
                index % 251 for index in range((1 << 20) + 17))

            def exchange_large():
                hosted.send_bytes(1, payload)
                self.assertEqual(hosted.receive_bytes(1, len(payload)), payload)

            large = threading.Thread(target=exchange_large)
            large.start()
            self.assertEqual(joined.receive_bytes(0, len(payload)), payload)
            joined.send_bytes(0, payload)
            large.join(timeout=10)
            self.assertFalse(large.is_alive())
        finally:
            hosted.close()
            joined.close()

    def test_tcp_rejects_duplicate_and_out_of_range_rank(self):
        port = _unused_port()
        outcome = {}

        def run_host():
            outcome["host"] = TCPTransport.host(
                rank=0, world_size=3, host="127.0.0.1", port=port, timeout=10.0)

        thread = threading.Thread(target=run_host)
        thread.start()
        with self.assertRaisesRegex(ValueError, "invalid transport rank"):
            TCPTransport.join(rank=3, world_size=3, host="127.0.0.1", port=port)
        results = {}
        barrier = threading.Barrier(2)

        def join_rank_one(name):
            barrier.wait(timeout=10)
            try:
                results[name] = TCPTransport.join(
                    rank=1, world_size=3, host="127.0.0.1", port=port,
                    timeout=10.0)
            except RuntimeError as error:
                results[name] = error

        racers = (
            threading.Thread(target=join_rank_one, args=("first",)),
            threading.Thread(target=join_rank_one, args=("second",)),
        )
        for racer in racers:
            racer.start()
        # Exactly one of the two duplicate rank-one registrations is accepted;
        # the other is rejected with a clear rendezvous error.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not any(
            isinstance(value, RuntimeError) for value in results.values()
        ):
            time.sleep(0.01)
        self.assertTrue(any(
            isinstance(value, RuntimeError)
            and "rank 1 is already registered" in str(value)
            for value in results.values()))
        rank_two = TCPTransport.join(
            rank=2, world_size=3, host="127.0.0.1", port=port, timeout=10.0)
        for racer in racers:
            racer.join(timeout=10)
            self.assertFalse(racer.is_alive())
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        rank_one = next(
            value for value in results.values()
            if isinstance(value, TCPTransport))
        hosted = outcome["host"]
        try:
            hosted.send(1, "host-to-one")
            self.assertEqual(rank_one.receive(0), "host-to-one")
            rank_two.send(0, "two-to-host")
            self.assertEqual(hosted.receive(2), "two-to-host")
        finally:
            hosted.close()
            rank_one.close()
            rank_two.close()

    def test_tcp_join_refused_names_the_address(self):
        port = _unused_port()
        with self.assertRaisesRegex(ConnectionError, f"127.0.0.1:{port}"):
            TCPTransport.join(
                rank=1, world_size=2, host="127.0.0.1", port=port,
                timeout=0.5)

    def test_tcp_join_handshake_timeout_is_a_clear_error(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        accepted = threading.Event()

        def silent_peer():
            connection, _address = listener.accept()
            accepted.set()
            time.sleep(2)
            connection.close()

        thread = threading.Thread(target=silent_peer, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                TCPTransport.join(
                    rank=1, world_size=2, host="127.0.0.1", port=port,
                    timeout=0.5)
            self.assertTrue(accepted.is_set())
        finally:
            listener.close()

    def test_tcp_two_rank_halo_forward_and_vjp(self):
        # Each destination reads itself and its two ring neighbors.
        entities = 8
        row_ptr = [3 * row for row in range(entities + 1)]
        col_idx = [
            source
            for destination in range(entities)
            for source in ((destination - 1) % entities, destination,
                           (destination + 1) % entities)
        ]
        port = _unused_port()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path = root / "ring.gfg"
            gf.save(
                gf.Graph.from_csr(
                    gf.tensor(row_ptr, dtype=gf.int64),
                    gf.tensor(col_idx, dtype=gf.int64),
                    num_src=entities, validate="full"),
                graph_path,
            )
            context = multiprocessing.get_context("spawn")
            processes = (
                context.Process(
                    target=_tcp_halo_worker,
                    args=(0, port, str(graph_path), directory)),
                context.Process(
                    target=_tcp_halo_worker,
                    args=(1, port, str(graph_path), directory)),
            )
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)
            records = [
                json.loads((root / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            # Matches the known-correct single-process ring result.
            self.assertEqual(records[0]["output"], [8.0, 3.0, 6.0, 9.0])
            self.assertEqual(records[1]["output"], [12.0, 15.0, 18.0, 13.0])
            self.assertTrue(all(
                record["gradient"] == [3.0, 3.0, 3.0, 3.0]
                for record in records))


if __name__ == "__main__":
    unittest.main()
