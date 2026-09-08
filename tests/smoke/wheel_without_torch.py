"""Installed-wheel smoke test in an environment that intentionally lacks Torch."""

from __future__ import annotations

import importlib.util
from importlib.metadata import metadata, version
from pathlib import Path
import subprocess
import sys

import tiga as gf


assert importlib.util.find_spec("torch") is None
assert "torch" not in sys.modules

distribution = metadata("tiga-lang")
assert distribution["Name"] == "tiga-lang"
assert version("tiga-lang") == gf.__version__
assert distribution["Maintainer"] == "walkerchi"
project_urls = set(distribution.get_all("Project-URL") or ())
assert "Repository, https://github.com/walkerchi/TIGA-lang.git" in project_urls
assert "Issues, https://github.com/walkerchi/TIGA-lang/issues" in project_urls

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
