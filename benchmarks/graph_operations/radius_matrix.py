"""Run the registered RadiusGraph lifecycle matrix and aggregate its gates.

Every leaf JSON is produced by ``radius_pipeline`` with matched semantics and
synchronized samples.  This driver is intentionally orchestration only; it
does not merge unlike lifecycle phases or invent speedups from medians.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.common.output_layout import operation_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--particles", type=int, default=1 << 15)
    parser.add_argument("--target-degree", type=float, default=32.0)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.particles = 1 << 11
        args.repeat = 5
    root = args.output_dir or operation_dir(
        "radius_distance_aggregation",
        f"matrix_{args.device}_n{args.particles}_degree{args.target_degree:g}",
    )
    root.mkdir(parents=True, exist_ok=True)

    cases = []
    failures = []
    for dimensions in (2, 3):
        for periodic in ("none", "box", "skew"):
            case_name = (
                f"{args.device}_n{args.particles}_d{dimensions}_"
                f"degree{args.target_degree:g}_{periodic}"
            )
            destination = root / case_name / "pipeline.json"
            command = [
                sys.executable, "-m",
                "benchmarks.graph_operations.radius_pipeline",
                "--device", args.device,
                "--particles", str(args.particles),
                "--dimensions", str(dimensions),
                "--target-degree", str(args.target_degree),
                "--periodic", periodic,
                "--repeat", str(args.repeat),
                "--json", str(destination),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode:
                failures.append({
                    "case": case_name,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                })
                continue
            payload = json.loads(destination.read_text())
            case_gates = payload["sota_gates"]
            cases.append({
                "case": case_name,
                "dimensions": dimensions,
                "periodic": periodic,
                "artifact": str(destination),
                "all_gates_passed": all(item["passed"] for item in case_gates),
                "gates": case_gates,
                "graph": payload["graph"],
            })
    all_passed = not failures and all(
        case["all_gates_passed"] for case in cases
    ) and len(cases) == 6
    summary = {
        "operation": "radius_distance_aggregation",
        "matrix_axes": {
            "dimensions": [2, 3],
            "periodic": ["none", "box", "skew"],
            "lifecycle": [
                "consume-only", "relation-reuse", "logical-rebind",
                "topology-rebuild+consume"
            ],
        },
        "config": {
            "device": args.device,
            "particles": args.particles,
            "target_degree": args.target_degree,
            "repeat": args.repeat,
        },
        "all_gates_passed": all_passed,
        "cases": cases,
        "failures": failures,
    }
    (root / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# RadiusGraph lifecycle matrix",
        "",
        f"Overall: **{'PASS' if all_passed else 'FAIL'}**",
        "",
        "| case | gate | matched peer | speedup | CI low | result |",
        "|---|---|---|---:|---:|---|",
    ]
    for case in cases:
        for gate in case["gates"]:
            lines.append(
                f"| {case['case']} | {gate['cache']} | {gate['baseline']} | "
                f"{gate['speedup_vs_sota']:.3f}x | "
                f"{gate['speedup_ci_low']:.3f}x | "
                f"{'PASS' if gate['passed'] else 'FAIL'} |"
            )
    for failure in failures:
        lines.append(f"\nBuild failure `{failure['case']}`: `{failure['stderr']}`")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(root / "matrix.json")
    print(root / "REPORT.md")
    if args.fail_on_gate and not all_passed:
        raise SystemExit("radius lifecycle matrix has failing or missing gates")


if __name__ == "__main__":
    main()
