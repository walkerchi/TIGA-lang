"""Neighborhood macro contracts and CPU/CUDA topology parity."""
from itertools import product
import subprocess
import sys

import pytest
import tiga as tg


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("kind", ["von_neumann", "moore"])
def test_macro_matches_distance_definition(ndim, radius, center, kind):
    macro = getattr(tg.stencil, kind)(radius, include_center=center)
    distance = lambda o: sum(map(abs, o)) if kind == "von_neumann" else max(map(abs, o))
    expected = tuple(o for o in product(range(-radius, radius + 1), repeat=ndim)
                     if distance(o) <= radius and (center or any(o)))
    assert macro.offsets(ndim) == expected


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("macro", [None, tg.stencil.von_neumann(2, include_center=False), tg.stencil.moore()])
def test_constructors_match_explicit_offsets(device, periodic, macro):
    import torch
    from tiga.interop.torch.graph import Graph as TorchGraph
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    offsets = (macro or tg.stencil.von_neumann()).offsets(2)
    expected = tg.Graph.stencil((2, 3), offsets, periodic=periodic, device=device)
    expected_rows = tuple(v.tolist() for v in expected.resolve_csr())
    for constructor in (tg.Graph, TorchGraph):
        actual = constructor.stencil((2, 3), macro, periodic=periodic, device=device)
        assert tuple(v.tolist() for v in actual.resolve_csr()) == expected_rows


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_invalid_radius(value):
    with pytest.raises(ValueError, match="radius"):
        tg.stencil.von_neumann(value)


def test_invalid_inputs():
    with pytest.raises(TypeError, match="include_center"):
        tg.stencil.moore(include_center=1)
    with pytest.raises(ValueError, match="kind"):
        tg.stencil.Neighborhood("unknown")
    with pytest.raises(ValueError, match="ndim"):
        tg.stencil.moore().offsets(0)
    for dims in ((), (True,), (0,), [3]):
        with pytest.raises(ValueError, match="dims"):
            tg.Graph.stencil(dims)
    with pytest.raises(ValueError, match="offsets"):
        tg.Graph.stencil((3,), ((True,),))


def test_no_torch_dependency():
    subprocess.run([sys.executable, "-c", """
import sys
sys.modules['torch'] = None
import tiga as tg
graph = tg.Graph.stencil((3, 3), tg.stencil.von_neumann())
assert graph.num_edges == 33
"""], check=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_documented_distance_examples(device, monkeypatch):
    import torch
    from pathlib import Path
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "examples"))
    from distance_metrics import relations
    graphs = relations(device)
    x = torch.tensor([[1., 0.], [.8, .6], [0., 1.], [-1., 0.]])
    euclidean = torch.cdist(x, x).fill_diagonal_(float("inf"))
    unit = torch.nn.functional.normalize(x, dim=-1)
    cosine = (1 - unit @ unit.T).fill_diagonal_(float("inf"))
    for graph, distances, cutoff in zip(graphs, [euclidean, euclidean, cosine, cosine], [.7, None, .25, None]):
        rows, columns = [v.tolist() for v in graph.resolve_csr()]
        for i in range(4):
            expected = (distances[i] <= cutoff).nonzero().flatten().tolist() if cutoff is not None else [distances[i].argmin().item()]
            assert columns[rows[i]:rows[i+1]] == expected
