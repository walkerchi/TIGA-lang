#!/usr/bin/env python3
"""Build the test-only utilities omitted by official LLVM binary SDKs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


DEFAULT_VERSION = "22.1.8"
DEFAULT_SHA256 = "922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()

    llvm_config = args.llvm_root / "bin" / "llvm-config"
    if not llvm_config.is_file():
        raise SystemExit(f"missing llvm-config: {llvm_config}")
    actual_version = subprocess.check_output(
        [str(llvm_config), "--version"], text=True).strip()
    if actual_version != args.version:
        raise SystemExit(
            f"LLVM SDK is {actual_version}, expected {args.version}")

    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.archive or (
        args.output / f"llvm-project-{args.version}.src.tar.xz")
    if not archive.exists():
        url = (
            "https://github.com/llvm/llvm-project/releases/download/"
            f"llvmorg-{args.version}/llvm-project-{args.version}.src.tar.xz"
        )
        temporary = archive.with_suffix(archive.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(archive)
    actual_digest = digest(archive)
    if actual_digest != args.sha256:
        raise SystemExit(
            f"source archive SHA256 is {actual_digest}, expected {args.sha256}")

    prefix = f"llvm-project-{args.version}.src/llvm/utils/"
    wanted_prefix = prefix + "lit/"
    wanted_files = {
        prefix + "FileCheck/FileCheck.cpp": "src/FileCheck.cpp",
        prefix + "not/not.cpp": "src/not.cpp",
        prefix + "count/count.c": "src/count.c",
        f"llvm-project-{args.version}.src/llvm/include/llvm/Support/AutoConvert.h":
            "include/llvm/Support/AutoConvert.h",
    }
    with tarfile.open(archive, "r:xz") as package:
        members = {member.name: member for member in package.getmembers()}
        for source_name, relative in wanted_files.items():
            member = members.get(source_name)
            if member is None or not member.isfile():
                raise SystemExit(f"source archive lacks {source_name}")
            target = args.output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = package.extractfile(member)
            if extracted is None:
                raise SystemExit(f"cannot extract {source_name}")
            with target.open("wb") as destination:
                shutil.copyfileobj(extracted, destination)
        for member in members.values():
            if not member.isfile() or not member.name.startswith(wanted_prefix):
                continue
            relative = Path(member.name).relative_to(wanted_prefix)
            target = args.output / "lit" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = package.extractfile(member)
            if extracted is None:
                raise SystemExit(f"cannot extract {member.name}")
            with target.open("wb") as destination:
                shutil.copyfileobj(extracted, destination)

    cxx = shutil.which("c++")
    if cxx is None:
        raise SystemExit("a C++ compiler is required")
    cxxflags = subprocess.check_output(
        [str(llvm_config), "--cxxflags"], text=True).split()
    ldflags = subprocess.check_output(
        [str(llvm_config), "--ldflags"], text=True).split()
    system_libs = subprocess.check_output(
        [str(llvm_config), "--system-libs"], text=True).split()
    bin_dir = args.output / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, components in (
        ("FileCheck", ("filecheck", "support")),
        ("not", ("support",)),
    ):
        libraries = subprocess.check_output(
            [str(llvm_config), "--libs", *components], text=True).split()
        run([
            cxx, *cxxflags, str(args.output / "src" / f"{name}.cpp"),
            "-o", str(bin_dir / name), *ldflags, *libraries, *system_libs,
        ])
    cc = shutil.which("cc")
    if cc is None:
        raise SystemExit("a C compiler is required")
    run([
        cc, str(args.output / "src" / "count.c"), "-O2",
        "-I", str(args.output / "include"), "-o",
        str(bin_dir / "count"),
    ])

    lit = args.output / "lit" / "lit.py"
    lit.chmod(lit.stat().st_mode | 0o111)
    run([str(bin_dir / "FileCheck"), "--version"])
    run([str(lit), "--version"])
    print(json.dumps({
        "llvm_version": actual_version,
        "filecheck": str(bin_dir / "FileCheck"),
        "not": str(bin_dir / "not"),
        "count": str(bin_dir / "count"),
        "lit": str(lit),
        "source_sha256": actual_digest,
    }, indent=2))


if __name__ == "__main__":
    main()
