"""Verify Graph.cu_seqlens and Graph.stencil against brute-force references."""

from __future__ import annotations

from itertools import pairwise

import tiga as tg
import pytest
import torch


def _reference_cu_seqlens(boundaries: list[int], causal: bool):
    rows = [0]
    columns: list[int] = []
    for start, end in pairwise(boundaries):
        for position in range(start, end):
            columns.extend(range(start, position + 1 if causal else end))
            rows.append(len(columns))
    return rows, columns


CU_SEQLENS_CASES = [
    [0, 3, 4, 9],           # varying lengths, including a length-1 sequence
    [0, 2, 2, 5, 5, 8],     # empty (length-0) sequences interleaved
    [0, 1, 2, 3, 4],        # all length-1 sequences
    [0, 7],                 # a single sequence
    [0],                    # no sequences at all
]


@pytest.mark.parametrize("family", ["native", "torch"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("empty", [False, True])
def test_transpose_bipartite_preserves_isolated_nodes(family, device, empty):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    factory = tg.tensor if family == "native" else torch.tensor
    dtype = tg.int64 if family == "native" else torch.int64
    graph = tg.Graph.from_csr(
        factory([0, 0, 0] if empty else [0, 2, 3], dtype=dtype, device=device),
        factory([] if empty else [0, 2, 2], dtype=dtype, device=device),
        num_src=4,
    )
    transposed = graph.transpose()
    assert transposed.schema.num_src == 2
    assert transposed.schema.num_dst == 4
    rows, columns = transposed.resolve_csr()
    assert rows.tolist() == ([0, 0, 0, 0, 0] if empty else [0, 1, 1, 3, 3])
    assert columns.tolist() == ([] if empty else [0, 0, 1])
    assert transposed.device == graph.device
    assert columns.dtype == tg.int64


@pytest.mark.parametrize("boundaries", CU_SEQLENS_CASES)
@pytest.mark.parametrize("causal", [True, False])
def test_cu_seqlens_matches_brute_force(boundaries, causal):
    expected_rows, expected_columns = _reference_cu_seqlens(boundaries, causal)
    graph = tg.Graph.cu_seqlens(
        torch.tensor(boundaries, dtype=torch.int64), causal=causal)
    assert graph.schema.num_src == boundaries[-1]
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == expected_rows
    assert col_idx.tolist() == expected_columns
    tg.Graph.from_csr(
        row_ptr, col_idx, num_src=boundaries[-1], validate="full")


@pytest.mark.parametrize("causal", [True, False])
def test_cu_seqlens_native_tensor_input(causal):
    boundaries = [0, 3, 4, 4, 8]
    expected_rows, expected_columns = _reference_cu_seqlens(boundaries, causal)
    graph = tg.Graph.cu_seqlens(
        tg.tensor(boundaries, dtype=tg.int64), causal=causal)
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == expected_rows
    assert col_idx.tolist() == expected_columns
    tg.Graph.from_csr(
        row_ptr, col_idx, num_src=boundaries[-1], validate="full")


def test_cu_seqlens_preserves_int32_index_dtype():
    graph = tg.Graph.cu_seqlens(torch.tensor([0, 2, 5], dtype=torch.int32))
    assert graph.schema.index_dtype.name == "int32"
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == [0, 1, 3, 4, 6, 9]
    assert col_idx.tolist() == [0, 0, 1, 2, 2, 3, 2, 3, 4]


def test_cu_seqlens_validation():
    with pytest.raises(ValueError, match="start at zero"):
        tg.Graph.cu_seqlens(torch.tensor([1, 2], dtype=torch.int64))
    with pytest.raises(ValueError, match="monotonic"):
        tg.Graph.cu_seqlens(torch.tensor([0, 3, 2], dtype=torch.int64))
    with pytest.raises(ValueError, match="one-dimensional"):
        tg.Graph.cu_seqlens(torch.zeros(2, 2, dtype=torch.int64))
    with pytest.raises(TypeError, match="causal"):
        tg.Graph.cu_seqlens(torch.tensor([0, 1], dtype=torch.int64), causal=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("causal", [True, False])
def test_cu_seqlens_builds_index_tensors_on_cuda(causal):
    boundaries = [0, 3, 4, 7]
    expected_rows, expected_columns = _reference_cu_seqlens(boundaries, causal)
    graph = tg.Graph.cu_seqlens(
        torch.tensor(boundaries, dtype=torch.int64, device="cuda"), causal=causal)
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == expected_rows
    assert col_idx.tolist() == expected_columns


def test_stencil_1d_three_point_truncates_boundaries():
    graph = tg.Graph.stencil((4,), ((-1,), (0,), (1,)))
    assert graph.schema.num_src == 4
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == [0, 2, 5, 8, 10]
    assert col_idx.tolist() == [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]
    assert graph.degree_bounds() == (2, 3)
    tg.Graph.from_csr(row_ptr, col_idx, num_src=4, validate="full")


def test_stencil_2d_five_point_exact():
    # Row-major (r, c) -> r * 3 + c on a 2x3 grid; offsets follow the
    # (0,0), (0,1), (0,-1), (1,0), (-1,0) order within each row.
    graph = tg.Graph.stencil(
        (2, 3), ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)))
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == [0, 3, 7, 10, 13, 17, 20]
    assert col_idx.tolist() == [
        0, 1, 3,
        1, 2, 0, 4,
        2, 1, 5,
        3, 4, 0,
        4, 5, 3, 1,
        5, 4, 2,
    ]
    tg.Graph.from_csr(row_ptr, col_idx, num_src=6, validate="full")


def test_stencil_periodic_1d_ring():
    graph = tg.Graph.stencil((4,), ((-1,), (0,), (1,)), periodic=True)
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == [0, 3, 6, 9, 12]
    assert col_idx.tolist() == [3, 0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 0]
    assert graph.fixed_degree() == 3
    tg.Graph.from_csr(row_ptr, col_idx, num_src=4, validate="full")


def test_stencil_validation():
    with pytest.raises(ValueError, match="dims"):
        tg.Graph.stencil((), ((0,),))
    with pytest.raises(ValueError, match="dims"):
        tg.Graph.stencil((0,), ((0,),))
    with pytest.raises(ValueError, match="offsets"):
        tg.Graph.stencil((3,), ())
    with pytest.raises(ValueError, match="offsets"):
        tg.Graph.stencil((3,), ((0, 0),))
    with pytest.raises(TypeError, match="periodic"):
        tg.Graph.stencil((3,), ((0,),), periodic=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_stencil_builds_index_tensors_on_cuda():
    graph = tg.Graph.stencil((4,), ((-1,), (0,), (1,)), device="cuda")
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == [0, 2, 5, 8, 10]
    assert col_idx.tolist() == [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]


CAT_LENGTHS = [
    [3, 1, 5],       # varying lengths, including a length-1 block
    [2, 0, 4],       # an empty block in the middle
    [1, 1, 1, 1],    # all length-1 blocks
    [7],             # a single block
]


@pytest.mark.parametrize("lengths", CAT_LENGTHS)
@pytest.mark.parametrize("causal", [True, False])
def test_cat_matches_cu_seqlens(lengths, causal):
    """cat of triangular/dense blocks is the cu_seqlens relation."""
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    block = tg.Graph.triangular if causal else tg.Graph.dense
    graph = tg.Graph.cat([block(length) for length in lengths])
    reference = tg.Graph.cu_seqlens(
        tg.tensor(boundaries, dtype=tg.int64), causal=causal)
    assert graph.schema.num_src == boundaries[-1]
    assert graph.schema.num_dst == boundaries[-1]
    row_ptr, col_idx = graph.resolve_csr()
    ref_rows, ref_columns = reference.resolve_csr()
    assert row_ptr.tolist() == ref_rows.tolist()
    assert col_idx.tolist() == ref_columns.tolist()
    tg.Graph.from_csr(
        row_ptr, col_idx, num_src=boundaries[-1], validate="full")


def test_cat_mixed_blocks_offsets_sources():
    dense_block = tg.Graph.dense(2, 3)  # num_src=2, num_dst=3
    csr_block = tg.Graph.from_csr(
        torch.tensor([0, 2, 3], dtype=torch.int64),
        torch.tensor([1, 2, 0], dtype=torch.int64),
        num_src=3)
    graph = tg.Graph.cat([dense_block, csr_block])
    assert graph.schema.num_src == 5
    assert graph.schema.num_dst == 5
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.tolist() == [0, 2, 4, 6, 8, 9]
    assert col_idx.tolist() == [0, 1, 0, 1, 0, 1, 3, 4, 2]


def test_cat_promotes_index_dtype_to_int64():
    graph = tg.Graph.cat(
        [tg.Graph.dense(2, index_dtype=tg.int32), tg.Graph.dense(2)])
    assert graph.schema.index_dtype.name == "int64"


def test_cat_validation():
    with pytest.raises(ValueError, match="non-empty"):
        tg.Graph.cat([])
    with pytest.raises(TypeError, match="list or tuple"):
        tg.Graph.cat(tg.Graph.dense(2))
    with pytest.raises(TypeError, match="only Graph"):
        tg.Graph.cat([tg.Graph.dense(2), 3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cat_builds_index_tensors_on_cuda():
    graph = tg.Graph.cat(
        [tg.Graph.triangular(4, device="cuda"),
         tg.Graph.dense(2, device="cuda")])
    row_ptr, col_idx = graph.resolve_csr()
    assert row_ptr.device.type == "cuda" or "cuda" in str(row_ptr.device)
    assert row_ptr.tolist() == [0, 1, 3, 6, 10, 12, 14]
    assert col_idx.tolist() == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5, 4, 5]
    with pytest.raises(ValueError, match="share a device"):
        tg.Graph.cat([tg.Graph.dense(2), tg.Graph.dense(2, device="cuda")])


def test_cat_result_feeds_message_passing():
    class Sum(tg.MessagePassing):
        reducer = tg.sum()

        def edge(self, src, dst, edge):
            return src.x

    x = tg.tensor([1.0, 2.0, 3.0, 4.0])
    graph = tg.Graph.cat([tg.Graph.dense(2), tg.Graph.dense(2)])
    out = Sum()(graph=graph, src={"x": x}, dst={})
    assert out.tolist() == [3.0, 3.0, 7.0, 7.0]
