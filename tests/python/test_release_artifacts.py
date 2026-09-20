"""Release tags and archive contents must agree exactly, not by substring."""
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from tools.verify_release import verify, verify_archive


def archives(tmp_path, version="0.1.0", metadata_version=None):
    content = f"Name: tiga-lang\nVersion: {metadata_version or version}\nRequires-Python: >=3.11,<3.13\n\n".encode()
    for abi in (311, 312):
        path = tmp_path / f"tiga_lang-{version}-cp{abi}-cp{abi}-manylinux_2_38_x86_64.whl"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(f"tiga_lang-{version}.dist-info/METADATA", content)
    with tarfile.open(tmp_path / f"tiga_lang-{version}.tar.gz", "w:gz") as z:
        info = tarfile.TarInfo(f"tiga_lang-{version}/PKG-INFO")
        info.size = len(content)
        z.addfile(info, io.BytesIO(content))


@pytest.mark.parametrize("tag", ["v0.1", "v1.0", "v0.1.0rc1", "0.1.0"])
def test_reject_substring_and_nonexact_tags(tmp_path, tag):
    with pytest.raises(ValueError, match="exactly"):
        verify(tmp_path, Path(__file__).resolve().parents[2] / "pyproject.toml", tag)


def test_verify_complete_matrix(tmp_path):
    archives(tmp_path)
    assert verify(tmp_path, Path(__file__).resolve().parents[2] / "pyproject.toml", "v0.1.0") == 3


def test_reject_metadata_mismatch_even_with_matching_filename(tmp_path):
    archives(tmp_path, metadata_version="9.0.0")
    with pytest.raises(ValueError, match="Version"):
        verify(tmp_path, Path(__file__).resolve().parents[2] / "pyproject.toml", "v0.1.0")


def test_release_workflow_requires_all_gates():
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github/workflows/release.yml").read_text()
    assert "needs: [quality, build-wheel, build-sdist, gpu-validation]" in source
    assert "tools/verify_release.py" in source
    assert "-m tools.gpu_gate" in source
    assert "source-dist/*.tar.gz" in source
    assert "expected not in name" not in source


def test_readthedocs_build_does_not_require_native_install():
    root = Path(__file__).resolve().parents[2]
    source = (root / ".readthedocs.yaml").read_text()
    assert "requirements: docs/requirements.txt" in source
    assert "path: ." not in source
    assert "fail_on_warning: true" in source


def test_first_release_matrix_and_notices_are_explicit():
    import tomllib
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.11,<3.13"
    assert not project["dependencies"]
    assert ".app.walkerchi.com" not in project["urls"]["Documentation"]
    for name in ("llvm-22.1.8.txt", "zstd-1.5.5.txt"):
        assert (root / "third_party/licenses" / name).stat().st_size > 1000


def test_pypi_readme_resolves_images_without_changing_repository_readme():
    import tomllib
    from tools.pypi_readme import dynamic_metadata

    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root/'pyproject.toml').read_text())
    assert 'readme' in config['project']['dynamic']
    assert config['tool']['dynamic-metadata'] == [
        {'provider': {'path': 'tools', 'module': 'pypi_readme'}},
    ]
    original = (root/'README.md').read_text()
    readme = dynamic_metadata({}, config['project'])['readme']
    assert readme['content-type'] == 'text/markdown'
    base = 'https://github.com/walkerchi/TIGA-lang/raw/refs/heads/main/'
    expected = original
    for image in ('assets/tiga-logo.png', 'docs/assets/compiler-performance-overview.svg'):
        expected = expected.replace(f'src="{image}"', f'src="{base}{image}"')
    assert readme['text'] == expected
    assert (root/'README.md').read_text() == original


def test_pypi_readme_preserves_remote_images_and_nonimage_links():
    from tools.pypi_readme import package_readme

    unchanged = '<img src="https://example.org/logo.png"><a href="README.zh.md">中文</a>'
    assert package_readme(unchanged, 'https://github.com/example/project.git') == unchanged
    assert package_readme("<img src='./assets/logo.png'>", 'https://github.com/example/project.git') == (
        "<img src='https://github.com/example/project/raw/refs/heads/main/assets/logo.png'>"
    )
