"""Regenerate every registered case image without rerunning benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.common.plotting import plot_latency, plot_roofline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/roofline"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("benchmarks/evidence_manifest.json"))
    parser.add_argument("--operation", action="append", default=[])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.operation)
    unknown = requested - set(manifest["operations"])
    if unknown:
        parser.error(f"unknown operations: {sorted(unknown)}")
    for operation, record in manifest["operations"].items():
        if requested and operation not in requested:
            continue
        for case in record.get("cases", []):
            directory = args.root / operation / case
            path = directory / record.get("artifact", "roofline.json")
            if not path.is_file():
                raise SystemExit(f"missing registered JSON: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "roof" in payload:
                plot_roofline(payload, directory)
                print(directory / "roofline.svg")
            plot_latency(payload, directory)
            print(directory / "provider_latency.svg")


if __name__ == "__main__":
    main()
