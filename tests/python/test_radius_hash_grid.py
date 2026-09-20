"""Hash-directory completeness, snapshot ownership and generated execution."""
from itertools import product

import pytest
import torch
import tiga as tg
from tiga.interop.torch.graph import from_native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Distance(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return edge.distance * src.x


def points(n, d, case):
    generator = torch.Generator(device="cuda").manual_seed(519)
    p = (torch.rand(n, d, generator=generator, device="cuda") * 64).floor()/64 - .5
    if case == "collisions":
        p[::2] += 64  # Same hash buckets, very different Euclidean positions.
    if case == "duplicates":
        p[1::2] = p[::2]
    return p


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("case", ["uniform", "collisions", "duplicates"])
def test_directory_has_exact_candidate_coverage(d, case):
    p = points(48, d, case)
    directory = tg.Graph.radius(p, .25).generated_cell_directory()
    assert directory.hash_grid and not directory.periodic
    ptr, order, coords = (v.cpu().tolist() for v in
                          (directory.cell_ptr, directory.particle_order, directory.cell_coordinates))
    side = int(directory.extents[0])
    assert sorted(order) == list(range(48))
    assert ptr[0] == 0 and ptr[-1] == 48
    assert ptr == sorted(ptr)
    distance = torch.cdist(p, p).cpu()
    for i, coordinate in enumerate(coords):
        candidates = []
        for offset in product((-1, 0, 1), repeat=d):
            cell = sum(((coordinate[a] + offset[a]) % side) * side**a for a in range(d))
            candidates.extend(order[ptr[cell]:ptr[cell+1]])
        assert len(candidates) == len(set(candidates))
        actual = sorted(j for j in candidates if j != i and distance[i, j] <= .25)
        expected = [j for j in range(48) if j != i and distance[i, j] <= .25]
        assert actual == expected


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("case", ["uniform", "collisions", "duplicates"])
def test_generated_forward_and_gradients(d, case, monkeypatch):
    monkeypatch.setenv("TIGA_RADIUS_CACHE_BYTES", "0")
    p = points(48, d, case).requires_grad_()
    x = torch.linspace(.1, 1., 48, device="cuda").requires_grad_()
    graph = tg.Graph.radius(p, .25)
    kernel = Distance()
    actual = kernel(graph=graph, src={"x": x}, dst={})
    # Selection is fixed; vector_norm gives zero derivative for duplicate points.
    delta = p[:, None, :] - p[None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1)
    mask = (distance <= .25) & ~torch.eye(48, device="cuda", dtype=torch.bool)
    expected = (distance * mask * x[None, :]).sum(-1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert "generated-radius" in kernel.last_variant.lowering
    assert "hash_grid = true" in kernel.ir("domain")
    assert from_native(graph)._radius_csr_cache is None
    actual_grad = torch.autograd.grad(actual.sum(), (p, x))
    expected_grad = torch.autograd.grad(expected.sum(), (p, x))
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e, atol=5e-5, rtol=5e-5)


def test_retained_snapshot_and_buffers_are_not_overwritten():
    p = points(48, 3, "uniform")
    view = from_native(tg.Graph.radius(p, .25))
    first = view.generated_cell_directory()
    first_ptr = first.cell_ptr.clone()
    p.add_(.4)
    second = view.generated_cell_directory()
    assert first.cell_ptr.data_ptr() != second.cell_ptr.data_ptr()
    torch.testing.assert_close(first.cell_ptr, first_ptr)
    # Borrowing an index tensor must keep that output slot unavailable too.
    borrowed = second.cell_ptr
    borrowed_copy = borrowed.clone()
    del second
    p.mul_(2)
    third = view.generated_cell_directory()
    assert borrowed.data_ptr() != third.cell_ptr.data_ptr()
    torch.testing.assert_close(borrowed, borrowed_copy)


def test_rebuild_reuses_unowned_slot_and_invalidates_snapshot():
    p = points(48, 3, "uniform")
    view = from_native(tg.Graph.radius(p, .25))
    first = view.generated_cell_directory()
    assert view.generated_cell_directory() is first
    scratch = view._hash_directory_pool[0]["keys"].data_ptr()
    del first
    p.add_(.25)
    second = view.generated_cell_directory()
    assert view._hash_directory_pool[0]["keys"].data_ptr() == scratch
    assert len(view._hash_directory_pool) == 1


def test_detached_directory_alias_survives_rebuild():
    p = points(48, 3, "uniform")
    view = from_native(tg.Graph.radius(p, .25))
    directory = view.generated_cell_directory()
    saved = directory.cell_ptr.detach()
    expected = saved.clone()
    del directory
    p.mul_(.1)
    rebuilt = view.generated_cell_directory()
    assert saved.data_ptr() != rebuilt.cell_ptr.data_ptr()
    torch.testing.assert_close(saved, expected)


def test_rebuild_changes_output_without_changing_compiled_plan(monkeypatch):
    monkeypatch.setenv("TIGA_RADIUS_CACHE_BYTES", "0")
    p = points(48, 3, "uniform")
    x = torch.ones(48, device="cuda")
    graph, kernel = tg.Graph.radius(p, .25), Distance()
    first = kernel(graph=graph, src={"x": x}, dst={}).clone()
    p.mul_(.5)
    second = kernel(graph=graph, src={"x": x}, dst={})
    distances = torch.cdist(p, p)
    expected = (distances * (distances <= .25)).sum(-1)
    torch.testing.assert_close(second, expected, atol=2e-5, rtol=2e-5)
    assert not torch.allclose(first, second)


def test_periodic_geometry_keeps_dense_directory():
    p = points(48, 3, "uniform")
    graph = tg.Graph.radius(p, .25, periodic=torch.ones(3, device="cuda"))
    directory = graph.generated_cell_directory()
    assert directory.periodic and not directory.hash_grid


def test_empty_and_unsupported_paths_remain_safe():
    assert tg.Graph.radius(torch.empty(0, 3, device="cuda"), .25).generated_cell_directory() is None
    graph = tg.Graph.radius(points(48, 3, "uniform"), .25,
                            metric=lambda src, dst, edge: edge.displacement.norm(dim=-1))
    assert graph.generated_cell_directory() is None
