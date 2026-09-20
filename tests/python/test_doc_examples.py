"""Keep the README and introductory documentation executable."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[2]
FENCES = re.compile(r"^([ \t]*)```python\n(.*?)^\1```", re.M | re.S)


def test_user_documentation_does_not_expose_internal_experiment_brief():
    paths = [ROOT / "mkdocs.yml", ROOT / "README.md", ROOT / "README.zh.md",
             *(ROOT / "docs").rglob("*.md")]
    forbidden = ("三个核心实验", "当前三组实验", "Three key experiments",
                 "three current experiments", "current question-led evaluation",
                 "图下方现在是", "本次未改默认路由", "旧环图和人为大 halo",
                 "不再作为主曲线", "私有 WireGuard", "Socket over private WireGuard",
                 "Earlier overlap probes remain", "平均分行也没有平衡",
                 "Earlier overlap measurements are retained", "旧 overlap 测量只保留",
                 "PROJECT.md §15.3", "ledger wins", "本轮验证摘要",
                 "NCCL/GPU-direct remains a separate validation gate",
                 "NCCL/GPU-direct 仍是独立门禁", "不是本次重测")
    for path in paths:
        for phrase in forbidden:
            assert phrase not in path.read_text(), (path, phrase)


def test_entry_pages_do_not_describe_internal_publication_work():
    for name in ("README.md", "README.zh.md", "docs/index.md", "docs/index.zh.md"):
        text = (ROOT / name).read_text()
        for phrase in ("private preview", "受登录保护", "官方发布目标", "First publication"):
            assert phrase not in text, (name, phrase)


def test_distributed_reference_matches_the_supported_transport_contract():
    for suffix in ("", ".zh"):
        api = (ROOT / f"docs/api{suffix}.md").read_text()
        roadmap = (ROOT / f"docs/roadmap{suffix}.md").read_text()
        assert "TCP/NCCL" in api and "TCP/NCCL" in roadmap
        assert "RCCL" in api
        assert "one-rank\n    NCCL fixture" not in api
        assert "单 rank 的 NCCL 测试夹具" not in api
        assert "CPU overlap" not in roadmap and "CPU 重叠" not in roadmap


def test_radius_example_uses_the_inclusive_cutoff_contract():
    for suffix in ("", ".zh"):
        text = (ROOT / f"docs/examples/solvers{suffix}.md").read_text()
        assert r"\lVert p_i - p_j \rVert \le r" in text
        assert r"\lVert p_i - p_j \rVert < r" not in text
    import xml.etree.ElementTree as ET
    root = ET.parse(ROOT / "docs/assets/examples/meshfree-radius.svg").getroot()
    circles = {c.attrib.get("class"): c.attrib for c in root.iter("{http://www.w3.org/2000/svg}circle")}
    for coord in ("cx", "cy"):
        assert circles["radius"][coord] == circles["query"][coord]


def _blocks(path: Path) -> list[str]:
    return [textwrap.dedent(m[2]) for m in FENCES.finditer(path.read_text())]


def _include(reference: str) -> str:
    filename, separator, section = reference.partition(":")
    path = ROOT / filename
    assert path.is_file(), f"missing snippet source: {reference}"
    source = path.read_text()
    if separator:
        start = f"# --8<-- [start:{section}]"
        end = f"# --8<-- [end:{section}]"
        assert source.count(start) == source.count(end) == 1, reference
        source = source.split(start, 1)[1].split(end, 1)[0]
    return source


def _run(path: str, monkeypatch, *, all_blocks: bool = False, block=None):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    blocks = _blocks(ROOT / path)
    assert blocks, f"no Python examples in {path}"
    code = blocks[block] if block is not None else "\n".join(blocks if all_blocks else blocks[:1])
    code = re.sub(
        r'^--8<-- "([^"]+)"$',
        lambda m: _include(m[1]),
        code,
        flags=re.M,
    )
    if 'torch.device("cuda")' in code:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("Documented GPU example requires CUDA; exercised by GPU release gate")
    namespace = {"__name__": "doc_example"}
    exec(compile(code, path, "exec"), namespace)
    return namespace


def _native(value, expected):
    assert value.tolist() == pytest.approx(expected)
    assert value.execution["backend"] == "cpu-llvm-jit"


def _torch(value, expected):
    import torch
    assert isinstance(value, torch.Tensor)
    assert value.tolist() == pytest.approx(expected)


def test_readme_example(monkeypatch):
    ns = _run("README.md", monkeypatch)
    _torch(ns["out"], [11.0, 8.0, 17.0])
    _torch(ns["dx"], [7.0, 10.0, 3.0])
    _torch(ns["dw"], [1.0, 3.0, 2.0, 1.0, 2.0])


@pytest.mark.parametrize("suffix", ["", ".zh"])
def test_graph_connection_walkthrough(suffix, monkeypatch):
    ns = _run(f"docs/programming-model{suffix}.md", monkeypatch, all_blocks=True)
    assert ns["sources"] == [0, 2]
    graph = ns["graph"]
    rows, columns = graph.resolve_csr()
    edges = [(columns.tolist()[edge], dst)
             for dst in range(graph.schema.num_dst)
             for edge in range(rows.tolist()[dst], rows.tolist()[dst + 1])]
    assert edges == [(0, 0), (2, 0), (1, 1), (0, 2), (1, 2)]


@pytest.mark.parametrize("path", ["README.md", "docs/getting-started.md", "docs/getting-started.zh.md"])
def test_installation_documents_pypi_and_source(path):
    source = (ROOT / path).read_text()
    assert "python -m pip install tiga-lang" in source
    assert 'python -m pip install "tiga-lang[cuda]"' in source
    assert "python -m pip install -e ." in source
    assert "python -m tiga" in source
    assert "pypi-install" in source

@pytest.mark.parametrize("suffix", ["", ".zh"])
@pytest.mark.parametrize("page", ["index", "programming-model", "message-passing", "runtime-and-autograd", "getting-started", "execution"])
def test_introductory_doc_examples(page, suffix, monkeypatch):
    ns = _run(f"docs/{page}{suffix}.md", monkeypatch,
              all_blocks=page == "getting-started")
    if page == "index":
        _torch(ns["out"], [11.0, 8.0, 17.0])
    elif page == "programming-model":
        _torch(ns["out"], [11.1, 8.2, 17.3])
        _torch(ns["dT"], [7.0, 10.0, 3.0])
        _torch(ns["dc"], [1.0, 3.0, 2.0, 1.0, 2.0])
        _torch(ns["db"], [1.0, 1.0, 1.0])
    elif page == "message-passing":
        _torch(ns["out"], [2.1, 1.2, 1.8])
        _torch(ns["dx"], [1.0, 1.0, 0.5])
        _torch(ns["dw"], [1.0, 3.0, 2.0, 1.0, 2.0])
    elif page == "runtime-and-autograd":
        _native(ns["loss"], 20.0)
        _native(ns["dx"], [2.0, 4.0, 6.0])
    elif page == "execution":
        _torch(ns["output"], [11.1, 8.2, 17.3])
        _torch(ns["d_temperature"], [7.0, 10.0, 3.0])
    else:
        _torch(ns["output"], [11.1, 8.2, 17.3])
        _torch(ns["d_temperature"], [7.0, 10.0, 3.0])
        _torch(ns["d_conductivity"], [1.0, 3.0, 2.0, 1.0, 2.0])
        _torch(ns["d_bias"], [1.0, 1.0, 1.0])


@pytest.mark.parametrize("suffix", ["", ".zh"])
@pytest.mark.parametrize("page", ["memory", "memory-and-distributed", "api-examples"])
def test_memory_and_reference_examples(page, suffix, monkeypatch):
    path = f"docs/{page}{suffix}.md"
    if page == "api-examples":
        for index in range(len(_blocks(ROOT / path))):
            _run(path, monkeypatch, block=index)
    else:
        _run(path, monkeypatch)


@pytest.mark.parametrize("suffix", ["", ".zh"])
def test_iter_walkthrough_numerical_loop(suffix, monkeypatch):
    namespace = _run(f"docs/ir-walkthrough{suffix}.md", monkeypatch, block=2)
    assert namespace["out"] == [23., 6., 0.]


def test_public_python_fences_have_valid_syntax():
    excluded = {"IR_DESIGN.md", "RELATED_WORK.md", "SCHEDULING_ABSTRACTIONS.md",
                "GPU_GRAPH_OPTIMIZATION.md", "design-notes.md"}
    for path in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
        if path.name in excluded or "rfcs" in path.parts:
            continue
        for index, code in enumerate(_blocks(path)):
            code = re.sub(r'^--8<-- "([^"]+)"$',
                          lambda m: textwrap.dedent(_include(m[1])), code, flags=re.M)
            ast.parse(code, filename=f"{path}:block-{index + 1}")


def test_python_import_alias_convention():
    """Keep the public Python alias distinct from the compiler's IR names."""
    sources = []
    for folder in ("examples", "benchmarks", "python", "tests", "tools"):
        sources.extend((str(p), p.read_text()) for p in (ROOT / folder).rglob("*.py")
                       if "__pycache__" not in p.parts)
    for path in [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]:
        sources.extend((str(path), code) for code in _blocks(path)
                       if "--8<--" not in code)
    for filename, source in sources:
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            # Historical design sketches are not runnable tutorials; the
            # syntax test above separately checks all public documentation.
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "tiga" and alias.asname is not None:
                        assert alias.asname == "tg", filename


def test_example_pages_have_titles_and_existing_source_links():
    for path in (ROOT / "docs" / "examples").glob("*.md"):
        source = path.read_text()
        assert re.search(r"^# [^#]", source, re.M), path
        for relative in re.findall(
            r"https://github\.com/walkerchi/TIGA-lang/blob/main/(examples/[^)#\s]+)",
            source,
        ):
            assert (ROOT / relative).is_file(), (path, relative)


def test_every_public_page_has_both_languages():
    excluded = {"IR_DESIGN.md", "RELATED_WORK.md", "SCHEDULING_ABSTRACTIONS.md",
                "GPU_GRAPH_OPTIMIZATION.md", "design-notes.md"}
    for path in (ROOT / "docs").rglob("*.md"):
        if path.name in excluded or "rfcs" in path.parts:
            continue
        counterpart = (path.with_name(path.name.replace(".zh.md", ".md"))
                       if path.name.endswith(".zh.md")
                       else path.with_suffix(".zh.md"))
        assert counterpart.is_file(), (path, counterpart)


def test_public_prose_uses_neutral_voice():
    excluded = {"IR_DESIGN.md", "RELATED_WORK.md", "SCHEDULING_ABSTRACTIONS.md",
                "GPU_GRAPH_OPTIMIZATION.md", "design-notes.md"}
    for path in (ROOT / "docs").rglob("*.md"):
        if path.name in excluded or "rfcs" in path.parts:
            continue
        source = re.sub(r"(?ms)^([ \t]*)```.*?^\1```", "", path.read_text())
        source = re.sub(r"https?://[^\s)]+|\{ #[^}]+\}|\]\([^)]*\)", "", source)
        assert not re.search(r"\b(?:you|your)\b|[你您]", source, re.I), path


def test_torch_examples_preserve_optional_dependency_and_native_type():
    import torch
    import tiga as tg

    metadata = (ROOT / "pyproject.toml").read_text()
    assert 'dependencies = []' in metadata.splitlines()
    # The default public examples use Torch, but explicit native constructors
    # retain their type: storage/control code must not silently change behavior.
    assert isinstance(tg.tensor([1.]), tg.Tensor)
    assert not isinstance(tg.tensor([1.]), torch.Tensor)
