"""Regenerate human-facing per-operation summaries from the formal manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.common.collection_plotting import (
    load_payloads,
    plot_compiler_report,
    plot_manifest_dashboard,
    plot_operation_summary,
    plot_release_showcase,
    write_interactive_compiler_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output/roofline"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("benchmarks/evidence_manifest.json"))
    parser.add_argument("--operation", action="append", default=[])
    parser.add_argument(
        "--report-output", type=Path,
        default=Path("docs/assets/compiler-performance-report.png"),
    )
    parser.add_argument(
        "--showcase-output", type=Path,
        default=Path("docs/assets/compiler-performance-overview.png"),
    )
    parser.add_argument(
        "--interactive-report-output", type=Path,
        default=Path("docs/assets/charts/compiler-performance-report.html"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.operation)
    unknown = requested - set(manifest["operations"])
    if unknown:
        parser.error(f"unknown operations: {sorted(unknown)}")
    generated = []
    for operation, record in manifest["operations"].items():
        if requested and operation not in requested:
            continue
        cases = record.get("cases", [])
        summary = args.root / operation / "summary.png"
        summary_svg = summary.with_suffix(".svg")
        summary_report = args.root / operation / "SUMMARY.md"
        if len(cases) < 2:
            summary.unlink(missing_ok=True)
            summary_svg.unlink(missing_ok=True)
            summary_report.unlink(missing_ok=True)
            continue
        paths = [args.root / operation / case / record.get("artifact", "roofline.json") for case in cases]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise SystemExit("missing registered JSON:\n" + "\n".join(missing))
        generated.append(plot_operation_summary(load_payloads(paths), args.root / operation))
    for path in generated:
        print(path)
    print(plot_manifest_dashboard(manifest, args.root))
    print(plot_compiler_report(
        manifest, args.root, args.report_output))
    print(plot_release_showcase(
        manifest, args.root, args.showcase_output))
    print(write_interactive_compiler_report(
        manifest, args.root, args.interactive_report_output))


if __name__ == "__main__":
    main()
