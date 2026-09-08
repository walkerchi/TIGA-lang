"""Canonical per-operation benchmark artifact layout."""

from __future__ import annotations

from pathlib import Path
import re


ROOFLINE_ROOT = Path("output/roofline")

OPERATIONS = {
    "weighted_aggregation": "CSR weighted sum / SpMV / SpMM",
    "diffusion": "CSR weighted source-minus-destination reduction",
    "radius_graph_build": "Dynamic radius neighbor construction",
    "radius_distance_aggregation": "Radius distance-weighted aggregation",
    "online_softmax": "Segmented/neighbor online softmax",
    "knn_graph": "Exact/approximate k-nearest-neighbor construction",
    "dense_matmul_calibration": "Dense compute roof calibration",
    "dense_attention": "Dense scaled dot-product attention",
    "linear_attention": "Linear attention with matched recurrent/chunk semantics",
    "sparse_attention": "Sparse attention with a matched mask and sparsity pattern",
    "message_passing_backward": "Compiler-generated MessagePassing reverse mode",
    "horizontal_fusion": "Fused multi-result CSR reduction",
    "cpu_pointwise_fusion": "CPU vector/parallel pointwise fusion",
    "cpu_relation": "CPU fused CSR relation traversal and reduction",
    "visualization_heatmap": "Fused Tensor-to-RGB heatmap raster preparation",
    "pagerank": "Fixed-iteration PageRank with fused CSR and node epilogue",
    "radius_edge_mlp": "Radius relation with edge-local MLP message",
    "edge_nn_backward": "Edge-local MLP training step with fused recompute VJP",
    "gat_attention": "GAT edge attention with fused online-softmax kernels",
}


def slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not result:
        raise ValueError("output case slug cannot be empty")
    return result


def operation_dir(operation: str, case: str = "default") -> Path:
    if operation not in OPERATIONS:
        raise ValueError(
            f"unknown roofline operation {operation!r}; "
            f"expected one of {sorted(OPERATIONS)}")
    return ROOFLINE_ROOT / operation / slug(case)


def artifact_path(
    operation: str,
    case: str,
    filename: str = "roofline.json",
) -> Path:
    path = operation_dir(operation, case) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
