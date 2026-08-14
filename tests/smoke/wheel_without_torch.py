"""Installed-wheel smoke test in an environment that intentionally lacks Torch."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import graphforge as gf


assert importlib.util.find_spec("torch") is None
assert "torch" not in sys.modules

x = gf.tensor([1.0, 2.0, 3.0], dtype=gf.float32)
y = ((x + 2.0) * x).sum()
assert float(y.tolist()) == 26.0

package = Path(gf.__file__).resolve().parent
gf_opt = package / "bin" / "gf-opt"
gf_translate = package / "bin" / "gf-translate"
assert gf_opt.is_file(), gf_opt
assert gf_translate.is_file(), gf_translate
version = subprocess.run(
    [str(gf_opt), "--version"], check=True, capture_output=True, text=True
).stdout
assert "22.1.8" in version, version
