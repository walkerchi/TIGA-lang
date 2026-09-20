# Dynamic and generated relations

Relations can be generated from coordinates or from fixed rules. “Generated”
does not necessarily mean the topology changes on every call. These examples
accept `torch.Tensor` directly, with no additional wrappers.

| Relation | What determines neighbors | Typical use |
|---|---|---|
| `Graph.stencil` | Grid dimensions and fixed offsets | Image filters and local grid updates |
| `Graph.radius` | Distances and a cutoff | Particles and spatial neighborhoods |
| `Graph.knn` | Distance ranking and k | Fixed-degree point-cloud computation |

## Five-point stencil on a regular grid { #regular-grid-stencil }

A [stencil](https://en.wikipedia.org/wiki/Stencil_(numerical_analysis)) specifies
fixed neighbor offsets. This example smooths a 2-D grid by averaging the center,
up, down, left and right values. No `positions` array or nearest-neighbor search
is needed.

![Five-point stencil: node 12 reads itself and neighbors 7, 11, 13 and 17. Only the local three-by-three neighborhood is shown.](../assets/examples/stencil-grid.svg)

`dims=(5, 5)` specifies five rows and five columns; offsets are `(row, column)`.
Nodes use row-major ordering: coordinate `(2, 2)` is node `12`, with neighbors
`7, 17, 11, 13`. The `tg.stencil.von_neumann()` macro includes the center by default.

### Neighborhood macros { #stencil-neighborhood-macros }

[Von Neumann](https://en.wikipedia.org/wiki/Von_Neumann_neighborhood) uses
[Manhattan distance](https://en.wikipedia.org/wiki/Taxicab_geometry); [Moore](https://en.wikipedia.org/wiki/Moore_neighborhood)
uses [Chebyshev distance](https://en.wikipedia.org/wiki/Chebyshev_distance). Both expand integer offsets with distance **at most**
`radius`, infer the dimension from `dims`, and include the center by default.
For integer Manhattan distance, `<= 1` is equivalent to `< 2`.

| Macro | 2-D offsets | Number including center |
|---|---|---|
| `tg.stencil.von_neumann()` | center and four axis neighbors | 5 |
| `tg.stencil.von_neumann(2)` | diamond of radius 2 | 13 |
| `tg.stencil.moore()` | full 3×3 neighborhood | 9 |

```python
graph = tg.Graph.stencil((64, 64), device="cuda")  # Default: von_neumann().
graph = tg.Graph.stencil((64, 64), tg.stencil.moore(), device="cuda")
four_neighbors = tg.stencil.von_neumann(include_center=False)
offsets = four_neighbors.offsets(2)  # ((-1, 0), (0, -1), (0, 1), (1, 0))
```

Explicit offset tuples remain supported. Macros expand in lexicographic order;
edge fields must follow that order after boundary filtering. On a small periodic
grid, different offsets can wrap to the same source: those edges remain distinct
and contribute separately. See [parameter reference](../api.md#stencilvon_neumann).

### Five-point averaging

$$
y_{r,c}=\frac{x_{r,c}+x_{r-1,c}+x_{r+1,c}+x_{r,c-1}+x_{r,c+1}}{5}
$$

```python
--8<-- "examples/stencil_message_passing.py:core"
```

`run("cuda")` returns the output and gradient, both shaped `(5, 5)`.
The center output is `12.0` and the top-left output is `6.0`; the gradient
of the total output sum is `1.0` for every input.

- `periodic=True` wraps across boundaries, so the top-left node still has five inputs.
- `periodic=False` omits out-of-grid neighbors; it does **not zero-pad**.
  With `tg.mean()`, a corner averages only its three valid inputs.
- Zero/constant padding is not implemented by this constructor. Omitting an
  edge is not equivalent to sending a zero-valued message, especially for `mean`.
- The current `Graph.stencil` constructor generates
  [CSR](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR_or_CRS)).
  A rule-based definition does not imply edge-free execution. Reuse the graph
  while dimensions and offsets remain unchanged.

Run [`python examples/stencil_message_passing.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/stencil_message_passing.py).
CUDA is the default; add `--device cpu` for a CPU-only environment.

??? example "Full source: examples/stencil_message_passing.py"

    ```python
    --8<-- "examples/stencil_message_passing.py"
    ```

The radius and kNN examples below instead depend on positions. Collapsed
compilation records describe particular captured runs; `kernel.explain()`
reports the actual execution plan.

## Euclidean and custom distances { #distance-metrics }

[Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) is the
default for both constructors. [Cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity)
is one minus cosine similarity; smaller distances mean closer neighbors.

| Constructor | Current distance support | Selection |
|---|---|---|
| `Graph.radius(x, cutoff=r)` | Euclidean by default; custom `metric` callable | distance ≤ cutoff |
| `Graph.knn(x, k=k)` | Euclidean only; no `metric` argument yet | k nearest candidates |

```python
--8<-- "examples/distance_metrics.py:core"
```

`metric(src, dst, edge)` receives batches of candidate pairs. `src.position`
and `dst.position` have shape `(P, D)`; extra `fields` appear on the endpoint
namespaces. Return a floating tensor of shape `(P,)` on the same device. The
metric callback receives `edge.displacement` (minimum-image displacement when
periodic); `edge.distance` is its result, not an input to the callback.
An optional `select(src, dst, edge)` runs afterward and can read `edge.distance`
and `edge.cutoff`; it returns a boolean `(P,)` tensor and only removes edges.
The example uses Torch callbacks with Torch coordinates; Torch-free native
callbacks must use native Tensor operations and currently require CPU.

The cosine radius example selects similarity ≥ 0.75, excludes self edges, and
connects only nodes 0 and 1 in both directions. Arbitrary custom metrics use
an **all-pairs correctness path**: the Euclidean cell-list bound no longer
applies. Even a callable that recomputes Euclidean distance takes this path;
omit `metric` to retain the built-in optimizations. No arbitrary-metric fused
performance claim is implied.

For cosine kNN, normalize **nonzero** vectors first:

$$
\|\hat{x}-\hat{y}\|_2^2 = 2\bigl(1-\cos(x,y)\bigr)
$$

The rankings agree, apart from numerical ties. Normalize both queries and
`candidates` for bipartite kNN. This changes the coordinates used by the graph;
it does not redefine any reported Euclidean distance as cosine distance.
Reject zero/near-zero vectors or define their behavior explicitly before using
this equivalence. Discrete neighbor selection is not differentiable; gradients
through supported selected-edge computations hold the neighbor set fixed.

Run [`python examples/distance_metrics.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/distance_metrics.py)
on CUDA, or append `--device cpu`. The tiny example explicitly materializes CSR
only to print and inspect the chosen neighbors.

## Differentiable radius relation { #differentiable-radius-relation }

**What it is.** A radius graph is built from point positions: two points are
connected whenever their Euclidean distance is at most a cutoff r, so the
topology is computed from the data rather than stored. On this graph the
example runs a message-passing sum — every node i collects, from each
neighbor j inside the cutoff, the neighbor's value `x[j]` weighted by the
distance between them. It then asks for the gradient of the output with
respect to the positions and the values: a vector–Jacobian product (VJP),
the backward pass of the same computation.

![Radius relation: p0 and p2 lie within the cutoff r of p1, so distance-weighted edges flow into p1; p0 and p2 are farther apart than r and get no edge](../assets/examples/radius-autograd.svg)

$$
\text{out}_i \;=\; \sum_{e\,=\,(j \to i),\; \lVert p_j - p_i \rVert \,\le\, r} \lVert p_j - p_i \rVert \cdot x_j
$$

The example uses Torch tensors and `torch.autograd.grad`. Geometry fields
are differentiable for the realized neighbor set; the discrete decision to add
or remove an edge at the cutoff is not differentiated.

```python
--8<-- "examples/radius_autograd.py:core"
```

??? example "Full source: examples/radius_autograd.py (runs as-is)"

    ```python
    --8<-- "examples/radius_autograd.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CPU"

        ```text
        ### DistanceWeightedSum
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: materialize-fixed-radius-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=1
        ```

    === "CUDA"

        ```text
        ### DistanceWeightedSum
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: materialize-fixed-radius-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=1
        ```

## Exact kNN feeding MessagePassing { #exact-knn-feeding-messagepassing }

**What it is.** k-nearest neighbors (kNN): given N points, find for each
point the k closest other points (here by squared Euclidean distance, self
excluded). The example turns each point's k neighbors into graph edges and
runs a message-passing sum over them: every point accumulates its neighbors'
values `x[j]` multiplied by an edge weight (all ones here, so the result is
a plain neighbor sum).

![kNN relation: one query point highlighted, arrows from its three nearest neighbors carry weighted messages into it; all other points are ignored](../assets/examples/knn-message-passing.svg)

$$
\text{out}_i \;=\; \sum_{j\,\in\,\mathrm{knn}(i,\,k)} w_{(j \to i)} \cdot x_j
$$

Exact kNN stays procedural: the current position snapshot is rebuilt lazily
and its fixed-k compressed sparse row (CSR) ABI is rebound to one generated
MessagePassing TTIR consumer. Selection and consume are separate contracts,
fused into a single launch — no CSR row pointer or selected-column tensor is
materialized.

```python
--8<-- "examples/knn_message_passing.py:core"
```

??? example "Full source: examples/knn_message_passing.py (runs as-is)"

    ```python
    --8<-- "examples/knn_message_passing.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CUDA"

        ```text
        ### NeighborSumTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-ranked-select-consume
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-ranked-relation, select-ranked-pairs-coordinate-hierarchy, select-candidate-tile, emit-local-stable-topk, emit-hierarchical-topk-merge, fuse-selected-edge-consumer, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        remark: [planning] exact candidate ranking and edge consumption share one launch
        remark: [planning] candidate tiles retain a power-of-two padded stable key state
        remark: [planning] no CSR row pointer or selected column tensor is materialized
        variant cache: hits=0, misses=1
        ```
