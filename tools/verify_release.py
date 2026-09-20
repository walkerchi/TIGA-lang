"""Check exact tag, source and archive metadata before any upload."""
import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import tomllib
import zipfile

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.specifiers import SpecifierSet


def verify_archive(path, *, name, version, requires_python):
    if path.name.endswith(".whl"):
        parsed_name, parsed_version, _, tags = parse_wheel_filename(path.name)
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(members) != 1:
                raise ValueError("wheel must contain exactly one METADATA")
            metadata = BytesParser().parsebytes(archive.read(members[0]))
    else:
        parsed_name, parsed_version = parse_sdist_filename(path.name)
        tags = set()
        with tarfile.open(path) as archive:
            members = [m for m in archive.getmembers() if m.name.count("/") == 1 and m.name.endswith("/PKG-INFO")]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError("sdist must contain exactly one root PKG-INFO")
            metadata = BytesParser().parsebytes(archive.extractfile(members[0]).read())
    if canonicalize_name(name) != parsed_name or str(parsed_version) != version:
        raise ValueError(f"filename does not match {name}=={version}: {path.name}")
    for field, expected in (("Name", name), ("Version", version), ("Requires-Python", requires_python)):
        actual = metadata[field]
        matches = (SpecifierSet(actual or "") == SpecifierSet(expected)
                   if field == "Requires-Python" else actual == expected)
        if not matches:
            raise ValueError(f"{path.name}: {field}={metadata[field]!r}, expected {expected!r}")
    return tags


def verify(directory, project_file, tag):
    project = tomllib.loads(project_file.read_text())["project"]
    if tag != "v" + project["version"]:
        raise ValueError(f"tag {tag!r} does not exactly match v{project['version']}")
    paths = sorted(directory.iterdir())
    if not paths or any(not p.is_file() or not (p.name.endswith(".whl") or p.name.endswith(".tar.gz")) for p in paths):
        raise ValueError("release directory must contain only wheels and a source archive")
    all_tags = set()
    for path in paths:
        all_tags.update(verify_archive(path, name=project["name"], version=project["version"], requires_python=project["requires-python"]))
    expected = {f"cp{abi}-cp{abi}-manylinux_2_38_x86_64" for abi in (311, 312)}
    if not expected.issubset({str(tag) for tag in all_tags}):
        raise ValueError("missing first-release Linux CPython 3.11/3.12 wheel matrix")
    if sum(p.name.endswith(".tar.gz") for p in paths) != 1:
        raise ValueError("release requires exactly one source archive")
    return len(paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--project", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    print(f"{verify(args.directory, args.project, args.tag)} release artifacts verified")
