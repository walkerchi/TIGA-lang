"""Bounded fixture ingestion and complete oracle checks before billion-edge runs."""
import hashlib

import numpy as np
import pytest
import tiga as tg

from benchmarks.memory_hierarchy.billion_edges import Average, build, footprint, oracle, values


def test_streamed_fixture_matches_explicit_csr(tmp_path):
    path = tmp_path/'ring.gfg'
    info = build(path, 31*16, chunk_rows=7)
    graph = tg.load(path)
    rows, columns = graph.resolve_csr()
    np.testing.assert_array_equal(rows.to_numpy(), np.arange(32)*16)
    expected_columns = (np.arange(31)[:, None]+np.arange(1, 17)) % 31
    np.testing.assert_array_equal(columns.to_numpy(), expected_columns.ravel())
    np.testing.assert_array_equal(graph.fields('src')['x'].to_numpy(), values(0, 31))
    for name, expected_hash in info['payload_sha256'].items():
        with (path/name).open('rb') as file:
            assert hashlib.file_digest(file, 'sha256').hexdigest() == expected_hash
    explicit = values(0, 31)[expected_columns].mean(axis=1)
    np.testing.assert_array_equal(oracle(0, 31, 31), explicit)
    with pytest.raises(FileExistsError):
        build(path, 31*16)


@pytest.mark.parametrize('edges,chunk', [(0, 4), (17, 4), (16, 0)])
def test_ingest_rejects_invalid_configuration(tmp_path, edges, chunk):
    with pytest.raises(ValueError):
        build(tmp_path/'invalid.gfg', edges, chunk_rows=chunk)
    assert not list(tmp_path.iterdir())


def test_billion_working_set_exceeds_16gib():
    sizes = footprint(1_000_000_000//16)
    assert sizes['resident_lower_bound_bytes'] == 24_500_000_008
    assert sizes['resident_lower_bound_bytes'] > 16*2**30
    assert sizes['output_bytes'] < 8*2**30


def test_forward_does_not_retain_host_output_pages(tmp_path, monkeypatch):
    from tiga.message_passing import paged
    monkeypatch.setenv('TIGA_TENSOR_BACKEND', 'native')
    path = tmp_path/'ring.gfg'
    build(path, 31*16, chunk_rows=7)
    drain = paged._drain_bytes

    class Payload(bytes):
        active = 0
        peak = 0

        def __new__(cls, value):
            result = super().__new__(cls, value)
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)
            return result

        def __del__(self):
            type(self).active -= 1

    monkeypatch.setattr(paged, '_drain_bytes', lambda output: Payload(drain(output)))
    graph = tg.load(path)
    with tg.execution(page_rows=7):
        output = Average()(graph=graph, src=graph.fields('src'), dst={}, prefetch=False)
    np.testing.assert_array_equal(output.to_numpy(), oracle(0, 31, 31))
    assert Payload.peak == 1
    assert Payload.active == 0
