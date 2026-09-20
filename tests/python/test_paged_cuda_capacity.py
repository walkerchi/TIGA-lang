"""Persistent fields are lazy, own their readers and stage real device pages."""
import gc

import numpy as np
import pytest
import tiga as tg


class Average(tg.MessagePassing):
    reducer = tg.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return aggregate * .25


def fixture(tmp_path):
    values = ((np.arange(1024)[:, None] % 97) + np.arange(4)[None, :]).astype(np.float32)
    graph = tg.Graph.stencil((1024,), ((1,), (2,), (3,), (4,)), periodic=True)
    path = tmp_path / 'graph.gfg'
    tg.save(graph, path, fields={'src': {'x': tg.tensor(values.tolist())}})
    return path, values


def test_escaped_fields_are_lazy_and_keep_reader(tmp_path):
    path, values = fixture(tmp_path)
    with tg.execution(memory={'ram': 0}):
        field = tg.load(path).fields('src')['x']
        gc.collect()
        assert field._buffer is None
    np.testing.assert_array_equal(field.to_numpy(), values)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_paged_recurrent_capacity(tmp_path, monkeypatch, device):
    monkeypatch.setenv('TIGA_TENSOR_BACKEND', 'native')
    if device.startswith('cuda'):
        try:
            tg.runtime.cuda_compute_capability()
        except RuntimeError:
            pytest.skip('CUDA unavailable')
    path, expected = fixture(tmp_path)
    tier = 'device' if device.startswith('cuda') else 'ram'
    with tg.execution(memory={tier: 48 << 10}, page_rows=32) as run:
        graph = tg.load(path, device=device)
        x = graph.fields('src')['x']
        assert str(x.device) == str(graph.device)
        for _ in range(20):
            x = Average()(graph=graph, src={'x': x}, dst={}, prefetch=False)
            expected = sum(np.roll(expected, -offset, axis=0) for offset in (1, 2, 3, 4)) * .25
            np.testing.assert_allclose(x.to_numpy(), expected, rtol=2e-6, atol=2e-6)
            assert run.memory_report()['peak_bytes'][tier] <= 48 << 10
        del x, graph
        gc.collect()
        assert run.memory_report()['live_bytes'][tier] == 0


def test_cuda_scatter_computed_source(monkeypatch):
    monkeypatch.setenv('TIGA_TENSOR_BACKEND', 'native')
    try:
        tg.runtime.cuda_compute_capability()
    except RuntimeError:
        pytest.skip('CUDA unavailable')
    x = tg.tensor([[1., 2.], [3., 4.]], device='cuda', requires_grad=True)
    destination = tg.tensor([1, 3], dtype=tg.int64, device='cuda')
    inverse = tg.tensor([-1, 0, -1, 1], dtype=tg.int64, device='cuda')
    y = (x * 2)._scatter_rows(destination, inverse, 4)
    np.testing.assert_array_equal(y.to_numpy(), [[0, 0], [2, 4], [0, 0], [6, 8]])
    np.testing.assert_array_equal(tg.autograd.grad(y.sum(), x).to_numpy(), [[2, 2], [2, 2]])


def test_cuda_large_vector_csr_and_empty_rows(monkeypatch):
    monkeypatch.setenv('TIGA_TENSOR_BACKEND', 'native')
    try:
        tg.runtime.cuda_compute_capability()
    except RuntimeError:
        pytest.skip('CUDA unavailable')
    from benchmarks.distributed.paper_scaling import native, case, MeanNeighbors
    rows, columns, values, expected = case(8192, 4)
    graph = tg.Graph.from_csr(native(rows, 'cuda'), native(columns, 'cuda'), num_src=8192)
    y = MeanNeighbors()(graph=graph, src={'x': native(values, 'cuda')}, dst={})
    np.testing.assert_array_equal(y.to_numpy(), expected)
    messages = tg.tensor([[1., 2.], [3., 4.], [5., 6.]], device='cuda')
    rows = tg.tensor([0, 0, 2, 3], dtype=tg.int64, device='cuda')
    y = messages.csr_segment_sum(rows, 3, degree_bounds=(0, 2))
    np.testing.assert_array_equal(y.to_numpy(), [[0, 0], [4, 6], [5, 6]])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('index_dtype', [tg.int32, tg.int64])
def test_paged_flat_indices_keep_empty_rows(tmp_path, monkeypatch, device, index_dtype):
    monkeypatch.setenv('TIGA_TENSOR_BACKEND', 'native')
    if device == 'cuda':
        try:
            tg.runtime.cuda_compute_capability()
        except RuntimeError:
            pytest.skip('CUDA unavailable')
    graph = tg.Graph.from_csr(
        tg.tensor([0, 0, 2, 2, 3], dtype=index_dtype),
        tg.tensor([0, 1, 2], dtype=index_dtype), num_src=3, validate='full')
    path = tmp_path/'empty-rows.gfg'
    tg.save(graph, path, fields={'src': {'x': tg.tensor([[1., 2.], [3., 4.], [5., 6.]])}})
    paged = tg.load(path, device=device)
    y = Average()(graph=paged, src=paged.fields('src'), dst={}, page_rows=1, prefetch=False)
    np.testing.assert_array_equal(y.to_numpy(), [[0, 0], [1., 1.5], [0, 0], [1.25, 1.5]])


def test_persistent_gather_and_vjp_without_numpy(tmp_path, monkeypatch):
    from tiga.graph import storage
    monkeypatch.setenv('TIGA_TENSOR_BACKEND', 'native')
    monkeypatch.setattr(storage, '_np', None)

    class Copy(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            return src.x

    graph = tg.Graph.stencil((4,), ((1,),), periodic=True)
    path = tmp_path/'numpy-free.gfg'
    tg.save(graph, path, fields={'src': {'x': tg.tensor([[1., 11.], [2., 12.], [3., 13.], [4., 14.]])}})
    graph = tg.load(path)
    x = graph.fields('src', requires_grad=True)['x']
    y = Copy()(graph=graph, src={'x': x}, dst={}, page_rows=1, prefetch=False)
    assert y.tolist() == [[2., 12.], [3., 13.], [4., 14.], [1., 11.]]
    assert tg.autograd.grad(y.sum(), x).tolist() == [[1., 1.]]*4
