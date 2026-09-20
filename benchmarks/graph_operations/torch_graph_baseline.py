"""Pure eager Torch exact builders with bounded distance workspace.

No Tiga/PyG, extension, index cache or approximate search. Materialize the
complete selected relation; chunk only distance rows, not the candidate domain.
"""
import torch

PAIR_BUDGET = 16 * 1024 * 1024


def distance_blocks(points, *, pair_budget=PAIR_BUDGET):
    """Yield direct squared distances, excluding the query itself."""
    n = len(points)
    if pair_budget < n:
        raise ValueError('pair_budget must fit one complete candidate row')
    rows = max(1, pair_budget // n)
    for start in range(0, n, rows):
        query = points[start:start+rows]
        distance = None
        for axis in range(points.shape[1]):
            delta = query[:, axis, None] - points[None, :, axis]
            delta.square_()
            if distance is None:
                distance = delta
            else:
                distance.add_(delta)
        local = torch.arange(len(query), device=points.device)
        distance[local, local+start] = float('inf')
        yield start, distance


def knn_indices(points, k, *, pair_budget=PAIR_BUDGET):
    """Exact all-candidate top-k, without a full N-by-N allocation."""
    output = torch.empty((len(points), k), device=points.device, dtype=torch.int64)
    for start, distance in distance_blocks(points, pair_budget=pair_budget):
        output[start:start+len(distance)] = distance.topk(k, largest=False).indices
    return output


def radius_csr(points, cutoff, *, pair_budget=PAIR_BUDGET):
    """Build all accepted edges and Euclidean weights using Torch only."""
    n = len(points)
    columns, weights, degrees = [], [], []
    for _, distance in distance_blocks(points, pair_budget=pair_budget):
        accepted = distance < cutoff * cutoff
        row, col = accepted.nonzero(as_tuple=True)
        columns.append(col)
        weights.append(distance[row, col].sqrt())
        degrees.append(accepted.sum(dim=1))
    row_ptr = torch.cat((torch.zeros(1, device=points.device, dtype=torch.int64),
                         torch.cat(degrees).cumsum(0)))
    return torch.sparse_csr_tensor(row_ptr, torch.cat(columns), torch.cat(weights),
                                   size=(n,n))
