"""Automatic budget-driven graph offload and prefetch-depth coverage."""

from __future__ import annotations

import pytest

import tiga as gf


class Smoothing(gf.MessagePassing):
    reducer = gf.sum()

    def edge(self, src, dst, edge):
        return src.x

    def node(self, dst, aggregate):
        return 0.25 * aggregate


_GRID = (60, 45)
_OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _reference():
    graph = gf.Graph.stencil(_GRID, _OFFSETS)
    assert graph.schema.realization == "materialized_csr"
    x = gf.tensor([float(node % 97) for node in range(graph.schema.num_dst)])
    return graph, x, Smoothing()(graph=graph, src={"x": x}, dst={}).tolist()


def test_budget_forces_paged_offload_and_matches(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    _graph, x, reference = _reference()
    with gf.runtime.auto_offload(ram=16 << 10):  # 16KB << the CSR
        graph = gf.Graph.stencil(_GRID, _OFFSETS)
        assert graph.schema.realization == "paged_csr"
        out = Smoothing()(graph=graph, src={"x": x}, dst={},
                          page_rows=700).tolist()
    assert out == reference


def test_under_budget_stays_materialized(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    with gf.runtime.auto_offload(ram=1 << 30):
        graph = gf.Graph.stencil(_GRID, _OFFSETS)
        assert graph.schema.realization == "materialized_csr"


def test_zero_budget_offloads_any_nonempty_csr(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    with gf.runtime.auto_offload(ram=0):
        graph = gf.Graph.stencil(_GRID, _OFFSETS)
        assert graph.schema.realization == "paged_csr"


def test_env_var_sets_default_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    monkeypatch.setenv("TIGA_GRAPH_RAM_BUDGET", "1024")
    graph = gf.Graph.stencil(_GRID, _OFFSETS)
    assert graph.schema.realization == "paged_csr"
    # The context manager wins over the environment variable.
    with gf.runtime.auto_offload(ram=1 << 30):
        graph = gf.Graph.stencil(_GRID, _OFFSETS)
        assert graph.schema.realization == "materialized_csr"


def test_invalid_budgets_are_rejected():
    with pytest.raises(ValueError):
        gf.runtime.auto_offload(ram=-1)
    with pytest.raises(ValueError):
        gf.runtime.auto_offload(ram="1GB")  # bytes only, no suffix parsing


def test_prefetch_depth_is_result_neutral(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    _graph, x, reference = _reference()
    with gf.runtime.auto_offload(ram=16 << 10):
        graph = gf.Graph.stencil(_GRID, _OFFSETS)
        for depth in (1, 2, 8):
            out = Smoothing()(graph=graph, src={"x": x}, dst={},
                              page_rows=431, prefetch_depth=depth).tolist()
            assert out == reference


def test_prefetch_depth_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("TIGA_SPILL_DIR", str(tmp_path))
    _graph, x, _ref = _reference()
    with gf.runtime.auto_offload(ram=16 << 10):
        graph = gf.Graph.stencil(_GRID, _OFFSETS)
        with pytest.raises(ValueError):
            Smoothing()(graph=graph, src={"x": x}, dst={}, prefetch_depth=0)
        monkeypatch.setenv("TIGA_PAGED_PREFETCH_DEPTH", "bogus")
        with pytest.raises(ValueError):
            Smoothing()(graph=graph, src={"x": x}, dst={}, page_rows=431)
