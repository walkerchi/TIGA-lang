"""Bind manual local-GPU approval to artifacts from a successful tagged build.

The digest identifies the locally tested wheel; it is a maintainer assertion,
not remote attestation that GPU tests ran. Keep the local test log separately.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def verify_evidence(directory, run, *, commit, repository, tag, gpu_wheel_sha256):
    if run.get("status") != "completed" or run.get("conclusion") != "success":
        raise ValueError("release build and all quality jobs must have succeeded")
    if run.get("path") != ".github/workflows/release.yml":
        raise ValueError("artifacts must come from the release workflow")
    for key in ("repository", "head_repository"):
        if run.get(key, {}).get("full_name", "").lower() != repository.lower():
            raise ValueError("release build repository does not match")
    if run.get("event") not in ("push", "workflow_dispatch"):
        raise ValueError("release build must not come from a pull request")
    if not tag.startswith("v") or run.get("head_branch") != tag:
        raise ValueError("release build must target the same version tag")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or run.get("head_sha") != commit:
        raise ValueError("release build commit does not match the version tag")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", gpu_wheel_sha256):
        raise ValueError("local GPU wheel SHA256 must contain exactly 64 hex digits")
    wheels = list(directory.glob("*-cp312-cp312-manylinux_2_38_x86_64.whl"))
    if len(wheels) != 1:
        raise ValueError("expected exactly one CPython 3.12 Linux GPU-tested wheel")
    with wheels[0].open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != gpu_wheel_sha256.lower():
        raise ValueError("release wheel differs from the locally GPU-validated wheel")
    return wheels[0].name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--gpu-wheel-sha256", required=True)
    args = parser.parse_args()
    wheel = verify_evidence(
        args.directory, json.loads(args.run_metadata.read_text()),
        commit=args.commit, repository=args.repository, tag=args.tag,
        gpu_wheel_sha256=args.gpu_wheel_sha256,
    )
    print(f"Successful tagged build and local GPU wheel digest verified: {wheel}")
