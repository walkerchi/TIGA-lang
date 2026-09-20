"""A disk-resident giant graph: save 1M nodes, stream pages, spill results."""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path

import tiga as tg


# --8<-- [start:core]
class Smoothing(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        del dst, edge
        return src.x  # staged edge item: neighbor value, (E_page,)

    def node(self, dst, aggregate):
        del dst
        return 0.25 * aggregate  # Jacobi average over the 4 neighbors, (N_page,)


def main(grid=(1000, 1000)):
    # Phase 1 — ingest: build the 1000x1000 four-neighbor stencil once and
    # persist it as a versioned .gfg (1M nodes, ~4M edges on NVMe).
    graph_path = Path(tempfile.mkdtemp()) / "giant.gfg"
    tg.save(
        tg.Graph.stencil(grid, ((-1, 0), (1, 0), (0, -1), (0, 1))),
        graph_path,
    )

    # Phase 2 — process: reopen the graph paged (manifest only; the CSR stays
    # on disk) and run the unchanged kernel. The executor streams destination
    # row pages of 100k rows, prefetching page k+1 while page k computes;
    # node/edge fields stay in RAM — only the topology is paged.
    paged = tg.load(graph_path)
    x = tg.tensor([float(node % 977) for node in range(paged.schema.num_dst)])
    started = time.perf_counter()
    out = Smoothing()(graph=paged, src={"x": x}, dst={})  # (1_000_000,)
    elapsed = time.perf_counter() - started

    # Phase 3 — write back: spill the result under a stable name and reattach
    # it the way another process would. The checksum survives the round trip.
    checksum = math.fsum(out.tolist())
    out.disk(name="giant-smoothed")
    reattached = tg.from_disk("giant-smoothed")  # lazy payload, loads on read
    assert math.fsum(reattached.tolist()) == checksum
    print(
        f"paged {paged.schema.num_dst:,} nodes / {paged.num_edges:,} edges "
        f"in {elapsed:.1f}s; checksum {checksum:.0f} round-tripped to disk")
# --8<-- [end:core]


# Inspect:
#   tg.load(graph_path).explain()            # realization=paged_csr, backing=nvme
#   Smoothing().variants[-1].remarks          # per-page native lowering record
#   TIGA_PAGED_PAGE_ROWS=250000 python examples/paged_giant_graph.py


if __name__ == "__main__":
    main()
