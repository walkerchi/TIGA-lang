"""Installed-wheel smoke test in an environment that intentionally lacks Torch."""

from __future__ import annotations

import importlib.util
from importlib.metadata import metadata, version
from email.utils import parseaddr
import os
from pathlib import Path
import subprocess
import sys

import tiga as tg


assert importlib.util.find_spec("torch") is None
assert "torch" not in sys.modules

# Tiny tensors otherwise use the Python oracle under the default auto policy.
# A wheel smoke must execute the bundled native compiler and runtime.
os.environ["TIGA_TENSOR_BACKEND"] = "native"

distribution = metadata("tiga-lang")
assert distribution["Name"] == "tiga-lang"
assert version("tiga-lang") == tg.__version__
assert parseaddr(distribution["Maintainer-email"]) == ("walkerchi", "walker.chi.000@gmail.com")
assert parseaddr(distribution["Author-email"]) == ("walkerchi", "walker.chi.000@gmail.com")
project_urls = set(distribution.get_all("Project-URL") or ())
assert "Repository, https://github.com/walkerchi/TIGA-lang.git" in project_urls
assert "Issues, https://github.com/walkerchi/TIGA-lang/issues" in project_urls

x = tg.tensor([1.0, 2.0, 3.0], dtype=tg.float32)
y = ((x + 2.0) * x).sum()
assert float(y.tolist()) == 26.0


# A MessagePassing forward + compiler-generated backward, exercised without
# Torch: the wheel must carry the full relation/autograd path, not just tensors.
class WeightedSum(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.weight * src.x


row_ptr = tg.tensor([0, 2, 3, 5], dtype=tg.int64)
col_idx = tg.tensor([0, 2, 1, 0, 1], dtype=tg.int64)
graph = tg.Graph.from_csr(row_ptr, col_idx, num_src=3, validate="full")
value = tg.tensor([1.0, 2.0, 3.0], requires_grad=True)
weight = tg.tensor([2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
out = WeightedSum()(graph=graph, src={"x": value}, dst={},
                    edge={"weight": weight})
assert out.tolist() == [11.0, 8.0, 17.0], out.tolist()
d_value, d_weight = tg.autograd.grad(out.sum(), (value, weight))
assert d_value.tolist() == [7.0, 10.0, 3.0], d_value.tolist()
assert d_weight.tolist() == [1.0, 3.0, 2.0, 1.0, 2.0], d_weight.tolist()
for result in (y, out, d_value, d_weight):
    assert result.execution["backend"] == "cpu-llvm-jit", result.execution
assert "torch" not in sys.modules

package = Path(tg.__file__).resolve().parent
gf_opt = package / "bin" / "gf-opt"
gf_translate = package / "bin" / "gf-translate"
assert gf_opt.is_file(), gf_opt
assert gf_translate.is_file(), gf_translate
for name in ("THIRD_PARTY_NOTICES.md", "llvm-22.1.8.txt", "zstd-1.5.5.txt"):
    assert (package / "licenses" / name).is_file(), name
version = subprocess.run(
    [str(gf_opt), "--version"], check=True, capture_output=True, text=True
).stdout
assert "22.1.8" in version, version
