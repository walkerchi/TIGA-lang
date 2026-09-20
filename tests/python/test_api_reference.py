"""Typed reference examples must execute and stay synchronized in both languages."""
import importlib.util
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("api_contracts", ROOT / "tools/render_api_reference.py")
contracts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contracts)


@pytest.mark.parametrize("anchor", [key for key, value in contracts.CONTRACTS.items() if value["runnable"]])
def test_reference_example(anchor, monkeypatch):
    monkeypatch.setenv("TIGA_TENSOR_BACKEND", "native")
    code = contracts.CONTRACTS[anchor]["example"]
    exec(compile(code, f"api-example:{anchor}", "exec"), {"__name__": "api_example"})


@pytest.mark.parametrize("lang,suffix", [("en", ""), ("zh", ".zh")])
def test_reference_is_current(lang, suffix):
    source = (ROOT / f"docs/api{suffix}.md").read_text()
    assert contracts.update(source, lang) == source
    anchors = set(re.findall(r"^### .*\{ #([^ }]+) \}", source, re.M))
    assert anchors == set(contracts.CONTRACTS)


def test_parameter_tables_cover_public_callable_signatures():
    import tiga as tg
    targets = {"gftensor": tg.tensor, "gfempty": tg.empty,
               "gffrom_torch": tg.from_torch, "tgexecution": tg.execution,
               "gfrepeat": tg.repeat, "gfwhile_loop": tg.while_loop,
               "gfnntrace": tg.nn.trace, "gfonline_softmax": tg.online_softmax,
               "stencilvon_neumann": tg.stencil.von_neumann,
               "stencilmoore": tg.stencil.moore,
               "stenciloffsets": tg.stencil.Neighborhood.offsets}
    for method in ("from_csr", "from_coo", "regular", "dense", "triangular",
                   "cu_seqlens", "cat", "stencil", "radius", "knn", "fields", "halo",
                   "paged_rows", "resolve_csr", "transpose", "explain"):
        targets["graph" + method] = getattr(tg.Graph, method)
    for method in ("reshape", "permute", "transpose", "squeeze", "unsqueeze",
                   "broadcast_to", "matmul", "exp", "sqrt", "conj", "cumsum",
                   "sum", "gather", "segment_sum", "checkpoint", "realize",
                   "prepare", "generated_code", "mlir", "to", "cpu", "spill"):
        targets["tensor" + method] = getattr(tg.Tensor, method)
    for method in ("heatmap", "particles", "delaunay", "mesh", "volume",
                   "load_ply", "load_obj", "export_ply", "export_obj", "export_vdb", "save_video"):
        targets["gfvisualize" + method.replace("_", "")] = getattr(tg.visualize, method)
    targets["tgvisualizegaussians"] = tg.visualize.gaussians
    for anchor, target in targets.items():
        documented = {part.strip().lstrip("*")
                      for name, *_ in contracts.CONTRACTS[anchor]["params"]
                      for part in name.split(" / ")}
        actual = set(inspect.signature(target).parameters) - {"self", "cls"}
        assert documented == actual, (anchor, actual - documented, documented - actual)
