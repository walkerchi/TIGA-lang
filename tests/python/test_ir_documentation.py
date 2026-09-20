"""Verify that the walkthrough's complete IR comes from the compiler fixture."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tiga.compiler.toolchain import find_gf_opt


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def gf_opt():
    tool = find_gf_opt()
    if tool is None:
        pytest.skip("IR documentation checks need built compiler tools")
    return tool


def test_ir_snapshots_match_compiler(gf_opt):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/render_ir_docs.py"),
         "--gf-opt", gf_opt, "--check"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 IR snapshots verified" in result.stdout


@pytest.mark.parametrize("stage", ["domain", "iter", "kernel", "task"])
def test_ir_snapshots_are_parseable_complete_modules(gf_opt, stage):
    result = subprocess.run(
        [gf_opt, str(ROOT / f"docs/includes/ir/{stage}.mlir")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "arith.mulf" in result.stdout
    assert 'input_roles = ["src", "edge"]' in result.stdout


def test_ir_snapshot_check_rejects_missing_files(gf_opt, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/render_ir_docs.py"),
         "--gf-opt", gf_opt, "--output-dir", str(tmp_path), "--check"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    assert "stale IR documentation" in result.stderr
    assert not list(tmp_path.iterdir())


def test_mlir_highlighting_preserves_text_and_marks_semantic_tokens():
    import importlib.util
    pygments = pytest.importorskip("pygments")
    from pygments.token import Name, Keyword, Comment, Error
    spec = importlib.util.spec_from_file_location("tiga_mlir_hook", ROOT / "docs/hooks/mlir_highlight.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    hook.on_startup()
    from pygments.lexers import get_lexer_by_name
    lexer = get_lexer_by_name("mlir")
    source = '%message = arith.mulf %source, %edge : f32 // edge\n'
    tokens = list(pygments.lex(source, lexer))
    assert "".join(value for _, value in tokens) == source
    assert (Name.Variable, "%message") in tokens
    assert (Name.Function, "arith.mulf") in tokens
    assert (Keyword.Type, "f32") in tokens
    assert (Comment.Single, "// edge") in tokens
    for stage in ("domain", "iter", "kernel", "task"):
        text = (ROOT / f"docs/includes/ir/{stage}.mlir").read_text()
        assert not any(token is Error for token, _ in pygments.lex(text, lexer))
