"""Inspect the billion-edge suite and map each profile to a memory tier.

This command is intentionally read-only: large datasets are never downloaded
implicitly.  The estimates are lower bounds for canonical CSR plus the named
algorithm state; builders, partitions and double-buffered I/O need additional
workspace and are reported separately by their eventual runners.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


GIB = 1 << 30
MANIFEST = Path(__file__).with_name("suite.json")


def load_suite(path: Path = MANIFEST) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        suite = json.load(stream)
    if suite.get("schema_version") != 1 or not isinstance(suite.get("cases"), list):
        raise ValueError("unsupported large-graph suite schema")
    return suite


def profile_bytes(case: dict, profile: dict) -> dict[str, int]:
    nodes = int(case["nodes"])
    entries = int(case["adjacency_entries"])
    column_bytes = 4 if nodes <= 0xFFFF_FFFF else 8
    row_offsets = (nodes + 1) * 8
    columns = entries * column_bytes
    edge_values = entries * int(profile["edge_value_bytes"])
    node_state = nodes * int(profile["node_state_bytes"])
    topology = row_offsets + columns
    return {
        "row_offsets": row_offsets,
        "column_indices": columns,
        "edge_values": edge_values,
        "node_state": node_state,
        "topology": topology,
        "minimum_working_set": topology + edge_values + node_state,
    }


def choose_tier(size: int, *, hbm: int, ram: int, ssd: int, headroom: float) -> str:
    if size <= hbm * headroom:
        return "hbm-resident"
    if size <= ram * headroom:
        return "ram-resident/hbm-streamed"
    if size <= ssd * headroom:
        return "ssd-resident/ram+hbm-streamed"
    return "distributed-or-more-storage"


def detect_capacity(ssd_path: Path) -> tuple[int, int, int]:
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    ssd = shutil.disk_usage(ssd_path).free
    hbm = 0
    try:
        import torch

        if torch.cuda.is_available():
            hbm = int(torch.cuda.get_device_properties(0).total_memory)
    except (ImportError, RuntimeError):
        pass
    return hbm, ram, ssd


def rows(suite: dict, *, hbm: int, ram: int, ssd: int, headroom: float):
    for case in suite["cases"]:
        for profile in case["profiles"]:
            sizes = profile_bytes(case, profile)
            yield {
                "case": case["id"],
                "profile": profile["name"],
                "nodes": case["nodes"],
                "adjacency_entries": case["adjacency_entries"],
                **sizes,
                "tier": choose_tier(
                    sizes["minimum_working_set"], hbm=hbm, ram=ram,
                    ssd=ssd, headroom=headroom),
            }


def _gib(value: int) -> str:
    return f"{value / GIB:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--ssd-path", type=Path, default=Path.cwd())
    parser.add_argument("--hbm-gib", type=float)
    parser.add_argument("--ram-gib", type=float)
    parser.add_argument("--ssd-gib", type=float)
    parser.add_argument("--headroom", type=float, default=0.8)
    parser.add_argument("--json", type=Path, help="write the capacity plan")
    args = parser.parse_args()
    if not 0 < args.headroom <= 1:
        parser.error("--headroom must be in (0, 1]")
    detected = detect_capacity(args.ssd_path)
    capacities = tuple(
        int(value * GIB) if value is not None else detected[index]
        for index, value in enumerate((args.hbm_gib, args.ram_gib, args.ssd_gib))
    )
    hbm, ram, ssd = capacities
    suite = load_suite(args.manifest)
    planned = list(rows(
        suite, hbm=hbm, ram=ram, ssd=ssd, headroom=args.headroom))
    print(
        f"capacity GiB: HBM={hbm / GIB:.1f} RAM={ram / GIB:.1f} "
        f"SSD-free={ssd / GIB:.1f}; usable={args.headroom:.0%}")
    print(
        "case                     profile                 nodes     entries  "
        "topology GiB  work GiB  target tier")
    for item in planned:
        print(
            f"{item['case']:<24} {item['profile']:<22} "
            f"{item['nodes']:>9,} {item['adjacency_entries']:>11,} "
            f"{_gib(item['topology']):>12} {_gib(item['minimum_working_set']):>9}  "
            f"{item['tier']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "manifest": str(args.manifest),
            "headroom": args.headroom,
            "capacity_bytes": {"hbm": hbm, "ram": ram, "ssd_free": ssd},
            "profiles": planned,
        }
        args.json.write_text(json.dumps(document, indent=2) + "\n")
        from benchmarks.common.diagnostic_plotting import plot_json
        plot_json(args.json)


if __name__ == "__main__":
    main()
