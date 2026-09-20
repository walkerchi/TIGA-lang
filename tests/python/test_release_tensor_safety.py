"""Public result ownership and inference-mode cache invalidation."""
from pathlib import Path
from importlib.machinery import EXTENSION_SUFFIXES

import pytest
import torch
import tiga as tg

from tiga.compiler.native import _compatible_extension
from tiga.interop.torch import provider
from tiga.interop.torch.state import tensor_version


class Weighted(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.w


class Distance(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x * edge.distance


@pytest.mark.parametrize("kind", ["empty", "row", "dense", "vector"])
@pytest.mark.parametrize("alias", ["detach", "dlpack", "storage"])
def test_public_output_helpers_preserve_storage_aliases(kind, alias):
    x = torch.zeros(3)

    def allocate(previous):
        if kind == "empty":
            return provider.reusable_empty_like(previous, x)
        if kind == "row":
            return provider.reusable_row_output(previous, x, rows=3)
        if kind == "vector":
            return provider.reusable_vector_output(previous, x, rows=3)
        return provider.reusable_dense_output(previous, x, lanes=1, rows=3, width=1)[0]

    previous = allocate(None)
    previous.fill_(7)
    if alias == "detach":
        saved = previous.detach()
    elif alias == "dlpack":
        saved = torch.from_dlpack(previous)
    else:
        saved = torch.empty(0).set_(previous.untyped_storage(), 0, previous.shape)
    next_output = allocate(previous)
    next_output.fill_(9)
    assert saved.data_ptr() != next_output.data_ptr()
    torch.testing.assert_close(saved, torch.full_like(saved, 7))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("relation", ["csr", "radius"])
@pytest.mark.parametrize("inference", [False, True])
def test_repeated_calls_preserve_detached_outputs_and_refresh_inputs(device, relation, inference):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with torch.inference_mode() if inference else torch.no_grad():
        x = torch.tensor([2., 3.], device=device)
        if relation == "csr":
            graph = tg.Graph.from_csr(torch.tensor([0, 1, 2], device=device),
                                      torch.tensor([1, 0], device=device), num_src=2)
            kernel = Weighted()
            args = dict(graph=graph, src={"x": x}, dst={}, edge={"w": torch.ones(2, device=device)})
        else:
            positions = torch.tensor([[0., 0., 0.], [.25, 0., 0.]], device=device)
            graph = tg.Graph.radius(positions, .5)
            kernel = Distance()
            args = dict(graph=graph, src={"x": x}, dst={})
        saved = kernel(**args).detach()
        expected = saved.clone()
        x.add_(1)
        new = kernel(**args)
        torch.testing.assert_close(saved, expected)
        torch.testing.assert_close(new, torch.tensor([4., 3.], device=device) * (.25 if relation == "radius" else 1))
        if relation == "radius":
            positions[1, 0] = 2
            torch.testing.assert_close(kernel(**args), torch.zeros_like(x))


def test_inference_tokens_never_validate_a_mutation_cache():
    x = torch.ones(2)
    assert tensor_version(x) == tensor_version(x)
    with torch.inference_mode():
        y = torch.ones(2)
        assert tensor_version(y) != tensor_version(y)


def test_extension_discovery_matches_complete_abi_names():
    for suffix in EXTENSION_SUFFIXES:
        assert _compatible_extension(Path("_tiga_compiler" + suffix))
    assert not _compatible_extension(Path("_tiga_compiler.cpython-999-x86_64-linux-gnu.so"))
    assert not _compatible_extension(Path("_tiga_compiler.backup.so"))


def test_editable_tool_discovery_follows_the_loaded_frontend(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from tiga.compiler import toolchain
    package = tmp_path / "python/tiga/compiler/toolchain.py"
    monkeypatch.setattr(toolchain, "__file__", str(package))
    monkeypatch.delenv("TIGA_OPT", raising=False)
    root = tmp_path / "build/current"
    extension = root / "python_bindings/_tiga_compiler.so"
    monkeypatch.setitem(sys.modules, "tiga._tiga_compiler", SimpleNamespace(__file__=str(extension)))
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    opt = bin_dir / "gf-opt"
    opt.touch()
    other = tmp_path / "build/newer/bin/gf-opt"
    other.parent.mkdir(parents=True)
    other.touch()
    assert toolchain.find_gf_opt() == str(opt)
    opt.unlink()
    assert toolchain.find_gf_opt() is None
