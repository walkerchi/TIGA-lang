"""Euclidean and cosine neighborhoods with ordinary Torch coordinates."""
import argparse

# --8<-- [start:core]
import torch
import torch.nn.functional as F
import tiga as tg


def cosine_distance(src, dst, edge):
    # One scalar per candidate pair. Inputs below have nonzero norms.
    return (1 - F.cosine_similarity(src.position, dst.position, dim=-1)).clamp_min(0)


def relations(device="cuda"):
    x = torch.tensor([[1., 0.], [.8, .6], [0., 1.], [-1., 0.]], device=device)
    euclidean_radius = tg.Graph.radius(x, cutoff=0.7)
    euclidean_knn = tg.Graph.knn(x, k=1)
    cosine_radius = tg.Graph.radius(x, cutoff=0.25, metric=cosine_distance)

    # On unit vectors, Euclidean and cosine distances have the same ranking.
    unit_x = F.normalize(x, dim=-1)
    cosine_knn = tg.Graph.knn(unit_x, k=1)
    return euclidean_radius, euclidean_knn, cosine_radius, cosine_knn
# --8<-- [end:core]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    for name, graph in zip(
        ("Euclidean radius", "Euclidean kNN", "cosine radius", "cosine kNN"),
        relations(args.device),
    ):
        # Materialize only to inspect this tiny example, not a production requirement.
        row_ptr, col_idx = graph.resolve_csr()
        print(f"{name}: row_ptr={row_ptr.tolist()}, col_idx={col_idx.tolist()}")
