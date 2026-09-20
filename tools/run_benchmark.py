"""Run a benchmark module while preserving fresh-run provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def capture(command):
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=15)
        return result.stdout.decode(errors="replace").strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--module", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.module.startswith("benchmarks.") or not all(part.isidentifier() for part in args.module.split(".")):
        parser.error("--module must name a benchmarks.* Python module")
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    command = [sys.executable, "-m", args.module, *arguments]
    packages = {}
    for name in ("numpy", "torch", "triton", "scipy", "matplotlib", "warp-lang", "torch-geometric"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True)
    untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard", "-z"],
                               cwd=ROOT, capture_output=True, check=True)
    extras = {}
    for name in untracked.stdout.decode().split("\0"):
        if name and (ROOT / name).is_file():
            extras[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    record = {"schema": "tiga.benchmark-run.v1", "command": command,
              "started_at": now(), "finished_at": None, "returncode": None,
              "git_revision": capture(["git", "rev-parse", "HEAD"]),
              "git_status": capture(["git", "status", "--short"]),
              "tracked_diff_sha256": hashlib.sha256(diff.stdout).hexdigest() if diff.returncode == 0 else None,
              "untracked_sha256": extras, "python": sys.version, "packages": packages,
              "platform": platform.platform(), "cpu": capture(["lscpu"]),
              "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
              "environment": {key: os.environ.get(key) for key in (
                  "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
                  "TIGA_TENSOR_BACKEND", "TIGA_OPT", "TIGA_TRANSLATE")},
              "seed": "Defined by module arguments/defaults; not inferred by wrapper",
              "scope": "Fresh invocation only; preserve module outputs and effective configuration separately"}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    try:
        result = subprocess.run(command, cwd=ROOT)
        record["returncode"] = result.returncode
        return result.returncode
    finally:
        record["finished_at"] = now()
        args.record.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
