"""Graph.halo automatically executes a rank-local MessagePassing program."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import tempfile

import tiga as gf
from tiga.distributed import DistributedRuntime, PipeTransport, owned_range


# --8<-- [start:core]
class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
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
# --8<-- [end:core]


# 8-node ring; rank r owns rows [4r, 4r+4). out[i] = x[i-1] + x[i] + x[i+1],
# and every node is read by exactly 3 rows, so the gradient is 3 everywhere —
# including the contributions that cross the rank boundary.
def main():
    entities = 8
    with tempfile.TemporaryDirectory() as directory:
        graph_path = Path(directory) / "ring.gfg"
        gf.save(
            gf.Graph.stencil(
                (entities,), ((-1,), (0,), (1,)), periodic=True),
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
    return shards

if __name__ == "__main__":
    main()
