"""Verify the portable chart input archive and optionally package it for docs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def verify(snapshot):
    index = json.loads((snapshot / "index.json").read_text())
    for item in index["files"]:
        path = snapshot / item["path"]
        if not path.resolve().is_relative_to(snapshot.resolve()):
            raise ValueError("evidence path escapes snapshot")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"checksum mismatch: {item['path']}")
        json.loads(raw)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "benchmarks/evidence_snapshot")
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    index = verify(args.snapshot)
    if args.zip:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            members = {"index.json": args.snapshot / "index.json",
                       "evidence_manifest.json": ROOT / "benchmarks/evidence_manifest.json"}
            members.update({item["path"]: args.snapshot / item["path"] for item in index["files"]})
            for name, source in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, source.read_bytes())
    print(f"{len(index['files'])} archived inputs verified")


if __name__ == "__main__":
    main()
