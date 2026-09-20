"""Archive registered chart inputs verbatim; never regenerate measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXTRA = (
    "roofline/radius_edge_mlp/quick/results.json",
    "roofline/radius_edge_mlp/default/results.json",
    "roofline/gat_attention/cuda_n131072_degree32/roofline.json",
    "roofline/cpu_relation/regular_permuted_i32_n16384_degree16_t16/roofline.json",
    "roofline/radius_distance_aggregation/cuda_n32768_d3_degree32/pipeline.json",
    "distributed/automatic_cpu_overlap/results.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "output")
    parser.add_argument("--destination", type=Path, default=ROOT / "benchmarks/evidence_snapshot")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "benchmarks/evidence_manifest.json").read_text())
    paths = set(EXTRA)
    for op, record in manifest["operations"].items():
        for case in record.get("cases", []):
            paths.add(f"roofline/{op}/{case}/{record.get('artifact', 'roofline.json')}")
    entries = []
    for relative in sorted(paths):
        source = args.source / relative
        payload = source.read_bytes()
        json.loads(payload)
        destination = args.destination / "output" / relative
        if args.check:
            if not destination.is_file() or destination.read_bytes() != payload:
                raise SystemExit(f"stale evidence snapshot: {relative}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        entries.append({"path": "output/" + relative, "sha256": hashlib.sha256(payload).hexdigest()})
    index = {"schema": "tiga.chart-evidence.v1", "files": entries,
             "provenance": "Historical local measurements; original revision/environment may be missing. Not measurements of the current checkout.",
             "render_command": "python -m benchmarks.common.plot_docs_results --root benchmarks/evidence_snapshot/output"}
    if not args.check:
        (args.destination / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"{len(entries)} benchmark input files {'verified' if args.check else 'archived'}")


if __name__ == "__main__":
    main()
