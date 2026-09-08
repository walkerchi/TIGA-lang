"""Execute the attention examples and verify results against reference oracles."""

from __future__ import annotations

import importlib.util
from itertools import pairwise
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

EXAMPLES = Path(__file__).parents[2] / "examples"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_attention_example_matches_sdpa():
    # The compiled dense-relation online softmax must match mask-free PyTorch
    # SDPA (exact same math).
    module = _load_module(
        "graphforge_example_full_attention", EXAMPLES / "full_attention.py")
    query, key, value = (  # (N, H, D) -> (H, N, D) for the SDPA oracle
        getattr(module, name).permute(1, 0, 2)
        for name in ("query", "key", "value"))
    expected = F.scaled_dot_product_attention(query, key, value)
    torch.testing.assert_close(
        module.output.permute(1, 0, 2), expected, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_causal_dense_relation_example_matches_causal_sdpa():
    # The triangular-graph causal reducer must match causal PyTorch SDPA.
    module = _load_module(
        "graphforge_example_causal_dense_relation",
        EXAMPLES / "causal_dense_relation.py")
    output, query, key, value = (  # example tensors are (N, H, D)
        item.permute(1, 0, 2) if item.ndim == 3 else item
        for item in module.main())
    expected = F.scaled_dot_product_attention(
        query, key, value, is_causal=True)
    torch.testing.assert_close(output, expected, rtol=4e-3, atol=4e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_varlen_causal_attention_example_matches_per_sequence_sdpa():
    # The packed cu_seqlens CSR (block-diagonal causal relation) runs on the
    # exact eager oracle; compare against per-sequence causal SDPA
    # concatenated along the packed axis.
    module = _load_module(
        "graphforge_example_varlen_causal",
        EXAMPLES / "varlen_causal_attention.py")
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(
                module.query[s:e].permute(1, 0, 2),     # (H, L, D)
                module.key[s:e].permute(1, 0, 2),
                module.value[s:e].permute(1, 0, 2),
                is_causal=True,
            ).permute(1, 0, 2)                          # (L, H, D)
            for s, e in pairwise(module.cu_seqlens)
        ]
    )
    torch.testing.assert_close(module.output, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tile_pruned_attention_example_output_is_well_formed():
    # The block-pruned approximate attention must produce a fully finite
    # (N, H, D) output.
    module = _load_module(
        "graphforge_example_tile_pruned_attention",
        EXAMPLES / "tile_pruned_attention.py")
    assert module.output.shape == (module.nodes, module.heads, module.width)
    assert torch.isfinite(module.output).all()
