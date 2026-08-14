"""Two-rank MPI integration helper launched by test_distributed_runtime.py."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mpi4py import MPI

import graphforge as gf
from graphforge.distributed import DistributedRuntime, owned_range


class NeighborSum(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x


def main() -> None:
    graph_path, result_path = Path(sys.argv[1]), Path(sys.argv[2])
    rank, world_size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    if world_size != 2:
        raise RuntimeError("MPI integration case requires exactly two ranks")
    graph = gf.load(graph_path).halo(gf.DeviceMesh("cpu", world_size), depth=1)
    begin, end = owned_range(graph.schema.num_dst, world_size, rank)
    local_x = gf.tensor(
        [float(entity) for entity in range(begin, end)], requires_grad=True)
    with DistributedRuntime.from_provider("mpi", communicator=MPI.COMM_WORLD):
        output = NeighborSum()(graph=graph, src={"x": local_x}, dst={})
        gradient = gf.autograd.grad(output.sum(), local_x)
        payload = {
            "rank": rank,
            "output": output.tolist(),
            "gradient": gradient.tolist(),
            "realization": graph.schema.realization,
        }
    (result_path / f"rank-{rank}.json").write_text(
        json.dumps(payload), encoding="utf-8")
    MPI.COMM_WORLD.Barrier()


if __name__ == "__main__":
    main()
