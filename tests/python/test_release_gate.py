"""A release gate must reject a green pytest run with missing coverage."""

from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("body,options,expected", [
    ("def test_ok(): pass", [], 0),
    ("import pytest\ndef test_missing(): pytest.skip('missing compiler')", [], 1),
    ("import pytest\npytest.importorskip('tiga_missing_gate_dependency')", [], 5),
    ("import pytest\n@pytest.mark.xfail\ndef test_broken(): assert False", [], 1),
    ("def test_ok(): pass\ndef test_excluded(): pass", ["-k", "test_ok"], 1),
])
def test_release_gate_requires_complete_execution(tmp_path, body, options, expected):
    case = tmp_path / "test_gate_case.py"
    case.write_text(body + "\n")
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        [sys.executable, "-c",
         "import pytest, sys; from tools.gpu_gate import ReleaseGate; "
         "sys.exit(pytest.main(sys.argv[1:], plugins=[ReleaseGate()]))",
         "-q", "-c", os.devnull, str(case), *options],
        cwd=Path(__file__).resolve().parents[2], env=env,
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    if expected:
        assert "GPU release gate incomplete" in result.stdout
