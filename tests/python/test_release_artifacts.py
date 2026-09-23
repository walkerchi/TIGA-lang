"""Release tags and archive contents must agree exactly, not by substring."""
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from tools.verify_release import verify, verify_archive


@pytest.fixture
def project_file(tmp_path):
    path = tmp_path / 'project.toml'
    path.write_text('[project]\nname = "tiga-lang"\nversion = "0.1.0"\nrequires-python = ">=3.11,<3.13"\n')
    return path


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
def test_reject_substring_and_nonexact_tags(tmp_path, project_file, tag):
    with pytest.raises(ValueError, match="exactly"):
        verify(tmp_path, project_file, tag)


def test_verify_complete_matrix(tmp_path, project_file):
    directory = tmp_path / 'dist'
    directory.mkdir()
    archives(directory)
    assert verify(directory, project_file, "v0.1.0") == 3


def test_reject_metadata_mismatch_even_with_matching_filename(tmp_path, project_file):
    directory = tmp_path / 'dist'
    directory.mkdir()
    archives(directory, metadata_version="9.0.0")
    with pytest.raises(ValueError, match="Version"):
        verify(directory, project_file, "v0.1.0")


def test_release_versions_are_consistent():
    import tomllib
    root = Path(__file__).resolve().parents[2]
    version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
    assert f'version: "{version}"' in (root / 'CITATION.cff').read_text()
    assert f'"{version}+source"' in (root / 'python/tiga/_version.py').read_text()
    assert f'gfrt_runtime_version(void) {{ return "{version}"; }}' in (root / 'lib/Runtime/Runtime.cpp').read_text()


def test_wheel_job_checks_out_same_tag_smoke_test():
    root = Path(__file__).resolve().parents[2]
    source = (root / '.github/workflows/release.yml').read_text()
    job = source.split('\n  build-wheel:', 1)[1].split('\n  build-sdist:', 1)[0]
    assert job.index('uses: actions/checkout@v4') < job.index('name: Smoke test')
    assert '"$GITHUB_WORKSPACE/tests/smoke/wheel_without_torch.py"' in job
    assert 'source-dist/*.tar.gz' in job


def test_release_workflow_requires_all_gates():
    root = Path(__file__).resolve().parents[2]
    source = (root / ".github/workflows/release.yml").read_text()
    assert 'uses: ./.github/workflows/compiler-ci.yml' in source
    assert 'gpu-validation:' not in source
    assert 'self-hosted' not in source
    assert 'gh-action-pypi-publish' not in source
    assert "source-dist/*.tar.gz" in source
    assert "expected not in name" not in source


def test_pypi_publication_requires_explicit_dispatch_not_a_tag_push():
    root = Path(__file__).resolve().parents[2]
    source = (root / '.github/workflows/publish.yml').read_text()
    assert "default: false" in source
    assert "if: inputs.publish_pypi && startsWith(github.ref, 'refs/tags/v')" in source
    assert 'workflow_dispatch:' in source
    assert '\n  push:' not in source
    assert 'python tools/verify_release_evidence.py dist' in source
    assert 'python tools/verify_release.py dist' in source
    assert source.count('run-id: ${{ inputs.build_run_id }}') == 2
    assert '-m build' not in source
    assert '-m pip wheel' not in source
    assert source.index('tools/verify_release_evidence.py') < source.index('gh-action-pypi-publish')


def test_native_ci_provisions_plotting_dependencies_and_disk_space():
    root = Path(__file__).resolve().parents[2]
    source = (root / '.github/workflows/compiler-ci.yml').read_text()
    assert 'pytest numpy pillow matplotlib packaging' in source
    assert source.index('Reserve disk space for LLVM') < source.index('Fetch and verify pinned LLVM SDK')
    assert 'rm -- "$archive"' in source


def test_native_ci_builds_wheel_and_mlir_tests_in_one_tree():
    root = Path(__file__).resolve().parents[2]
    source = (root / '.github/workflows/compiler-ci.yml').read_text()
    assert '-Cbuild-dir=build/ci' in source
    assert '-Ccmake.define.TIGA_INCLUDE_TESTS=ON' in source
    assert '-Cbuild.targets=all -Cbuild.targets=check-tiga' in source
    assert 'cmake -S . -B build/ci' not in source
    assert 'cmake --build build/ci' not in source


def test_release_builds_overlap_quality_but_publication_still_waits():
    root = Path(__file__).resolve().parents[2]
    source = (root / '.github/workflows/release.yml').read_text()
    wheel_job = source.split('\n  build-wheel:', 1)[1].split('\n  build-sdist:', 1)[0]
    assert 'needs: build-sdist' in wheel_job
    assert 'needs: [quality, build-sdist]' not in wheel_job
    assert 'source-dist/*.tar.gz' in wheel_job
    assert 'echo "CCACHE_BASEDIR=$RUNNER_TEMP" >> "$GITHUB_ENV"' in wheel_job
    assert 'echo "TMPDIR=$RUNNER_TEMP" >> "$GITHUB_ENV"' in wheel_job
    assert 'uses: ./.github/workflows/compiler-ci.yml' in source


@pytest.mark.parametrize('workflow', ['compiler-ci.yml', 'release.yml'])
def test_native_workflows_cache_pinned_sdk_and_validate_compiler_content(workflow):
    root = Path(__file__).resolve().parents[2]
    source = (root / '.github/workflows' / workflow).read_text()
    assert 'key: llvm-22.1.8-linux-x64-df0e1ecf' in source
    assert "if: steps.llvm-cache.outputs.cache-hit != 'true'" in source
    assert 'CMAKE_C_COMPILER_LAUNCHER: ccache' in source
    assert 'CMAKE_CXX_COMPILER_LAUNCHER: ccache' in source
    assert 'CCACHE_COMPILERCHECK: content' in source
    assert 'CCACHE_SLOPPINESS' not in source
    assert 'CMAKE_BUILD_PARALLEL_LEVEL: "2"' in source
    assert 'ccache --show-stats' in source


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
