"""Inventory vendor plugins and emit reproducible external hardware gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tiga.codegen import discover_providers, get_provider, provider_conformance


TARGETS = {
    "rocm-dcu": {"devices": ("hip", "dcu"), "artifact": "hsaco", "input_ir": "ttir"},
    "metal": {"devices": ("metal", "mps"), "artifact": "metallib", "input_ir": "msl"},
    "ppu": {"devices": ("ppu",), "artifact": "vendor-binary", "input_ir": "vendor-ir"},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/providers/conformance/results.json"),
    )
    parser.add_argument("--require", action="append", default=[])
    args = parser.parse_args()
    installed = discover_providers()
    results = {}
    for target, requirement in TARGETS.items():
        candidates = []
        for name in installed:
            provider = get_provider(name)
            report = provider_conformance(provider)
            if set(report["devices"]) & set(requirement["devices"]):
                candidates.append(report)
        valid = [
            report for report in candidates
            if report["input_ir"] == requirement["input_ir"]
            and requirement["artifact"] in report["artifacts"]
            and report["supports_async"]
        ]
        results[target] = {
            "status": "READY_FOR_HARDWARE" if valid else "PENDING_EXTERNAL_PLUGIN",
            "requirement": requirement,
            "providers": valid,
            "hardware_gates": [
                "correctness", "artifact-inspection", "cold-warm-compile",
                "roofline", "matched-provider-performance",
            ],
        }
    document = {
        "schema": "tiga.provider-conformance.v1",
        "abi_version": 1,
        "installed_providers": installed,
        "targets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    from benchmarks.common.diagnostic_plotting import plot_json
    plot_json(args.output)
    print(json.dumps(document, indent=2))
    missing = [
        target for target in args.require
        if results.get(target, {}).get("status") != "READY_FOR_HARDWARE"
    ]
    if missing:
        raise SystemExit("required providers unavailable: " + ", ".join(missing))


if __name__ == "__main__":
    main()
