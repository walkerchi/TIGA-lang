"""Graph.halo automatically executes a rank-local MessagePassing program."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import tempfile

import graphforge as gf
from graphforge.distributed import DistributedRuntime, PipeTransport, owned_range


class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


def worker(rank, endpoint, graph_path, queue):
    # The handle is a normal Graph. Only this rank's CSR rows are read from
    # the .gfg PhysicalInstance; topology is not eagerly loaded process-wide.
    graph = gf.load(graph_path).halo(gf.DeviceMesh("cpu", 2), depth=1)
    entities = graph.schema.num_dst
    begin, end = owned_range(entities, 2, rank)
    # Each process owns only its local leading-dimension shard. MessagePassing
    # performs exact halo pack/exchange/unpack below the kernel call.
    local_x = gf.tensor(
        [float(entity) for entity in range(begin, end)], requires_grad=True)
    transport = PipeTransport(rank, 2, {1 - rank: endpoint})
    with DistributedRuntime(transport):
        output = NeighborSum()(graph=graph, src={"x": local_x}, dst={})
        gradient = gf.autograd.grad(output.sum(), local_x)
        values, gradients = output.tolist(), gradient.tolist()
    queue.put((rank, (values, gradients)))


def main():
    entities = 8
    row_ptr = [3 * row for row in range(entities + 1)]
    col_idx = [
        source
        for destination in range(entities)
        for source in (
            (destination - 1) % entities,
            destination,
            (destination + 1) % entities,
        )
    ]
    with tempfile.TemporaryDirectory() as directory:
        graph_path = Path(directory) / "ring.gfg"
        gf.save(
            gf.Graph.from_csr(
                gf.tensor(row_ptr, dtype=gf.int64),
                gf.tensor(col_idx, dtype=gf.int64),
                num_src=entities, validate="full"),
            graph_path,
        )
        context = multiprocessing.get_context("spawn")
        first, second = context.Pipe(duplex=True)
        queue = context.Queue()
        processes = (
            context.Process(target=worker, args=(0, first, graph_path, queue)),
            context.Process(target=worker, args=(1, second, graph_path, queue)),
        )
        for process in processes:
            process.start()
        shards = dict(queue.get(timeout=10) for _ in processes)
        for process in processes:
            process.join(timeout=10)
            if process.exitcode:
                raise RuntimeError(f"rank exited with code {process.exitcode}")
    print("rank-local (output, gradient):", shards)


if __name__ == "__main__":
    main()
