"""Paged disk-resident giant-graph benchmark: bounded RSS and prefetch overlap.

One invocation measures one configuration in a fresh process — peak RSS
(``getrusage``) is a process-lifetime maximum, so configurations must not
share a process. A driver runs this script once per configuration, e.g.::

    python benchmarks/memory_hierarchy/paged_giant_graph.py --grid 2000 --build-only
    for cfg in "4000000 0" "100000 0" "100000 1"; do
        python benchmarks/memory_hierarchy/paged_giant_graph.py --grid 2000 $cfg
    done

The point: peak RSS tracks the page size, not the graph size, and prefetch
hides the disk reads behind computation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import tiga as gf


class Smoothing(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return 0.25 * aggregate


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=2000,
                        help="grid is grid×grid nodes, ~4 edges per node")
    parser.add_argument("--page-rows", type=int, default=100_000)
    parser.add_argument("--prefetch", type=int, default=1, choices=(0, 1))
    parser.add_argument("--build-only", action="store_true",
                        help="persist the graph cache and exit")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/memory_hierarchy/paged_giant_graph"),
    )
    args = parser.parse_args()

    graph_path = args.cache_dir / f"graphforge_paged_bench_{args.grid}.gfg"
    if not graph_path.exists():
        started = time.perf_counter()
        gf.save(
            gf.Graph.stencil(
                (args.grid, args.grid), ((-1, 0), (1, 0), (0, -1), (0, 1))),
            graph_path,
        )
        print(f"ingest {args.grid}x{args.grid}: "
              f"{time.perf_counter() - started:.1f}s")
    if args.build_only:
        return

    graph = gf.load(graph_path)
    x = gf.tensor(
        [float(node % 977) for node in range(graph.schema.num_dst)])
    started = time.perf_counter()
    output = Smoothing()(graph=graph, src={"x": x}, dst={},
                         page_rows=args.page_rows,
                         prefetch=bool(args.prefetch))
    elapsed = time.perf_counter() - started
    checksum = sum(output.tolist())

    row = {
        "grid": [args.grid, args.grid],
        "nodes": graph.schema.num_dst,
        "edges": graph.num_edges,
        "page_rows": args.page_rows,
        "prefetch": bool(args.prefetch),
        "pages": -(-graph.schema.num_dst // args.page_rows),
        "elapsed_s": round(elapsed, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        "checksum": round(checksum, 3),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"pr{args.page_rows}_pf{args.prefetch}.json"
    destination.write_text(json.dumps(row, indent=2) + "\n")
    print(f"page_rows={row['page_rows']:>10,} prefetch={row['prefetch']} "
          f"pages={row['pages']:>4} elapsed={row['elapsed_s']:7.1f}s "
          f"peak_rss={row['peak_rss_bytes'] / 2**20:8.0f}MB "
          f"checksum={row['checksum']:.0f}")


if __name__ == "__main__":
    main()
