"""Numerical checks for the introductory stencil, FEM and Gaussian examples."""
import importlib.util
from pathlib import Path
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'examples' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_stencil_matches_torch_roll(device):
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    result, gradient = load('stencil_message_passing').run(device)
    x = torch.arange(25, dtype=torch.float32, device=device).reshape(5, 5).requires_grad_()
    expected = (x + x.roll(1, 0) + x.roll(-1, 0) + x.roll(1, 1) + x.roll(-1, 1)) / 5
    expected_gradient, = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(gradient, expected_gradient)


def test_minimal_fem_matches_analytic_solution(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'examples'))
    module = load('fem_poisson_minimal')
    assert module.error < 1e-6
    assert module.u.tolist()[3] == pytest.approx(0.125, abs=1e-6)
    assert max(abs(a - b) for a, b in zip(module.apply(module.u).tolist(), module.b.tolist())) < 1e-6


def test_native_stencil_topology_copies_once_into_cached_torch_view():
    import torch
    import tiga as tg
    from tiga.interop.torch.graph import from_native
    graph = tg.Graph.stencil((3,), ((-1,), (0,), (1,)))
    view = from_native(graph)
    assert from_native(graph) is view
    rows, columns = view.resolve_csr()
    torch.testing.assert_close(rows, torch.tensor([0, 2, 5, 7]))
    torch.testing.assert_close(columns, torch.tensor([0, 1, 0, 1, 2, 1, 2]))


def test_torch_owned_topology_keeps_zero_copy():
    import torch
    import tiga as tg
    from tiga.graph.core import Graph
    from tiga.interop.torch.graph import from_native
    rows, columns = torch.tensor([0, 1, 2]), torch.tensor([1, 0])
    graph = Graph.from_csr(tg.from_torch(rows), tg.from_torch(columns), num_src=2)
    actual_rows, actual_columns = from_native(graph).resolve_csr()
    assert actual_rows.data_ptr() == rows.data_ptr()
    assert actual_columns.data_ptr() == columns.data_ptr()


def test_gaussian_scene_is_reproducible_and_rotations_are_tangent():
    from tiga.visualize.splats import _quaternion_to_matrix
    module = load('visualize_gaussians')
    scene = module.torus_gaussians(120)
    for first, second in zip(scene, module.torus_gaussians(120)):
        np.testing.assert_array_equal(first, second)
    positions, colors, scales, rotations, opacities = scene
    matrix = _quaternion_to_matrix(rotations)
    angle = np.arctan2(positions[:, 1], positions[:, 0])
    expected = np.stack([-np.sin(angle), np.cos(angle), np.zeros(len(angle))], axis=1)
    np.testing.assert_allclose(matrix[:, :, 0], expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-12)
    assert np.all(np.abs(positions) + 3 * scales.max(axis=1)[:, None] < 1)
    assert np.all((colors >= 0) & (colors <= 1))
    density = module.splat_density(positions, opacities, scales, voxel=32)
    assert np.isfinite(density).all() and density.max() > 0
    assert np.count_nonzero(density[[0, -1]]) == 0


def test_voxel_density_matches_one_rotated_gaussian():
    module = load('visualize_gaussians')
    position = np.array([[0.4, 0.4, 0.0]])
    scales = np.array([[0.12, 0.04, 0.03]])
    grid = module.splat_density(position, np.array([0.6]), scales, voxel=21)
    # At x=.3, y=.5 the displacement is purely tangent to the ring.
    expected = 0.6 * np.exp(-0.5 * (np.sqrt(0.02) / 0.12)**2)
    assert grid[13, 15, 10] == pytest.approx(expected)


def test_gaussian_example_exports_images_and_gif(tmp_path, monkeypatch):
    import sys
    from PIL import Image
    monkeypatch.setattr(sys, 'argv', ['visualize_gaussians.py', '--count', '24',
        '--size', '32', '--frames', '2', '--output', str(tmp_path)])
    load('visualize_gaussians').main()
    for name in ('gaussians_splats.png', 'gaussians_volume.png'):
        with Image.open(tmp_path / name) as picture:
            assert picture.size == (32, 32)
    with Image.open(tmp_path / 'gaussians_orbit.gif') as animation:
        assert animation.n_frames == 2


@pytest.mark.parametrize('suffix', ['', '.zh'])
def test_refreshed_documentation_uses_shared_sources(suffix):
    docs = ROOT / 'docs/examples'
    stencil = (docs / f'dynamic-relations{suffix}.md').read_text()
    fem = (docs / f'solvers{suffix}.md').read_text()
    visualization = (docs / f'visualization{suffix}.md').read_text()
    assert 'examples/stencil_message_passing.py:core' in stencil
    assert 'periodic=False' in stencil and 'periodic=True' in stencil
    assert 'examples/fem_poisson_minimal.py:core' in fem
    assert 'examples/visualize_gaussians.py:core' in visualization
    assert 'examples/visualize_gaussians.py:volume' in visualization
    assert '<video controls' in visualization and 'gaussians_orbit.mp4' in visualization
    video_tag = re.search(r'<video\b[^>]*>', visualization).group()
    assert 'autoplay' not in video_tag
