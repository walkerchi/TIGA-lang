# Attention and dense relations

Attention-style workloads expressed with generic relation and reducer
semantics. The mask is never code — it is the graph: a dense relation is full
attention, a triangular relation is causal attention, and a block-diagonal CSR
is varlen causal attention with `cu_seqlens`. Tile pruning is a separate,
approximate semantic layered on the same online-softmax reducer.

- [`python examples/full_attention.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/full_attention.py)
- [`python examples/causal_dense_relation.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/causal_dense_relation.py)
- [`python examples/varlen_causal_attention.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/varlen_causal_attention.py)
- [`python examples/tile_pruned_attention.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/tile_pruned_attention.py)

### Dense and causal { #dense-and-causal }

## Full attention, no mask { #full-attention-no-mask }

**What it is.** Attention: each of the N positions carries a query vector, a
key vector and a value vector of width D. Every output position is a weighted
average of all value vectors, where the weights are a softmax over query·key
similarity scores — so the cost is the full N×N score matrix. With no mask,
every query attends to every key. In graph terms each edge (j→i) pairs
destination position i (the query) with source position j (the key and
value); a dense relation contains all N×N edges.

![An 8×8 attention matrix with every cell filled: rows are query positions, columns are key positions, and a filled cell is an edge j→i — full attention has all N² edges, so there is nothing to mask.](../assets/examples/full-attention.svg)

$$s_{hij} = \langle k_{hj},\, q_{hi}\rangle \cdot D^{-1/2}, \qquad
\mathrm{out}_{hi} = \sum_{j} \frac{e^{s_{hij}}}{\sum_{l} e^{s_{hil}}}\, v_{hj}$$

This is the exact special case of everything below — the same tiled
online-softmax machinery FlashAttention uses, with every tile admitted.

```python
--8<-- "examples/full_attention.py:core"
```

??? example "Full source: examples/full_attention.py (runs as-is)"

    ```python
    --8<-- "examples/full_attention.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CUDA"

        ```text
        ### FullAttentionTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-dense-streaming-reduction
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-structured-message, expand-reducer-regions, select-cartesian-coordinate-hierarchy, select-dense-query-neighbor-tile, lower-vector-contraction, lower-streaming-reducer-state, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        accepted: [machine-schedule] admitted dense-query-key-tile
        unknown: [pipeline] no compiler-controlled asynchronous pipeline admitted
        remark: [planning] Cartesian relation remained implicit
        remark: [planning] multi-value message and reducer state were fused into tiles
        remark: [planning] no pairwise score or normalized-weight tensor was materialized
        executable cache: hits=0, misses=1
        ```

## Causal dense relation { #causal-dense-relation }

**What it is.** Causal attention is attention with a look-ahead ban:
position i may average only over positions j ≤ i. This is the rule in
language models that generate text left to right, where each position may
only read the past. The example expresses the mask as the graph itself: a
triangular relation contains exactly the edges (j→i) with j ≤ i, so the
edge function needs no masking code at all. As above, source j supplies
the key and value, destination i supplies the query.

![The same attention matrix with only the lower triangle filled (j ≤ i): the upper-triangle cells are not masked-out scores but edges that do not exist.](../assets/examples/causal-dense-relation.svg)

$$s_{hij} = \langle k_{hj},\, q_{hi}\rangle \cdot D^{-1/2} \quad (j \le i), \qquad
\mathrm{out}_{hi} = \sum_{j \le i} \frac{e^{s_{hij}}}{\sum_{l \le i} e^{s_{hil}}}\, v_{hj}$$

`Graph.triangular()` carries the lower-inclusive topology through
Domain → Iter → Kernel, and an ordinary online reducer lowers to causal
streaming TTIR. Torch appears only as external CUDA storage.

```python
--8<-- "examples/causal_dense_relation.py:core"
```

??? example "Full source: examples/causal_dense_relation.py (runs as-is)"

    ```python
    --8<-- "examples/causal_dense_relation.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CUDA"

        ```text
        ### CausalWeightedValueTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-dense-streaming-reduction
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-structured-message, expand-reducer-regions, select-cartesian-coordinate-hierarchy, select-dense-query-neighbor-tile, lower-vector-contraction, lower-streaming-reducer-state, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        accepted: [machine-schedule] admitted dense-query-key-tile
        unknown: [pipeline] no compiler-controlled asynchronous pipeline admitted
        remark: [planning] Cartesian relation remained implicit
        remark: [planning] multi-value message and reducer state were fused into tiles
        remark: [planning] no pairwise score or normalized-weight tensor was materialized
        executable cache: hits=0, misses=1
        ```

## Varlen causal attention with cu_seqlens { #varlen-causal-attention-with-cu_seqlens }

**What it is.** Real training batches pack many sequences of different
lengths into one flat tensor and mark the boundaries with cumulative lengths
`cu_seqlens = [0, L₁, L₁+L₂, …]` (the convention of flash-attn's varlen API).
Each sequence attends causally **within itself only** — position i may read
position j only when both lie in the same sequence and j ≤ i. As a mask that
is a block-diagonal matrix of small triangles; as a graph it is one explicit
CSR relation with exactly those edges, Σ L(L+1)/2 of them. Boundary handling
needs no code: cross-sequence attention is an edge that does not exist.

![A 5×5 attention matrix for cu_seqlens = [0, 3, 5]: two block-diagonal causal triangles — positions 0–2 attend within sequence 0 and positions 3–4 within sequence 1; every cross-sequence cell has no edge.](../assets/examples/varlen-causal-attention.svg)

$$\mathrm{out}_{hi} = \sum_{\substack{j \le i \\ \mathrm{seq}(j)=\mathrm{seq}(i)}}
\frac{e^{s_{hij}}}{\sum_{\substack{l \le i \\ \mathrm{seq}(l)=\mathrm{seq}(i)}} e^{s_{hil}}}\, v_{hj},
\qquad
\mathrm{seq}(i) = b \;\Leftrightarrow\; \mathrm{cu}_b \le i < \mathrm{cu}_{b+1}$$

The example builds the block-diagonal causal CSR in one line —
`gf.Graph.cu_seqlens(cu_seqlens, causal=True)` — and runs
the same unmasked edge UDF, including a length-1 sequence. The same relation
can equally be spelled as a composition of per-sequence blocks,
`gf.Graph.cat([gf.Graph.triangular(L) for L in lengths])`: block `k` attends
only within itself, so the two constructors produce the identical CSR. (The tests check
all three attention examples against `F.scaled_dot_product_attention`.)
Execution falls back to the
exact eager oracle today: compiled streaming covers the dense/triangular
relations above and field-score CSR segment softmax, while an edge-computed
score over a general CSR is a compiler boundary, not a semantic one.

```python
--8<-- "examples/varlen_causal_attention.py:core"
```

??? example "Full source: examples/varlen_causal_attention.py (runs as-is)"

    ```python
    --8<-- "examples/varlen_causal_attention.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CUDA"

        ```text
        ### VarlenCausalAttentionTorchExecutor
        backend: reference
        provider: torch
        lowering: (unspecified)
        passes: validate-domain, select-reference-csr
        remark: [planning] reference evaluator selected; no performance codegen artifact exists
        remark: [planning] cross-apply fusion is handled by the @gf.jit/@gf.program capture boundary rather than the single-apply reference evaluator
        executable cache: hits=0, misses=1
        ```

### Sparse and pruned { #sparse-and-pruned }

## Tile-pruned sparse attention { #tile-pruned-sparse-attention }

**What it is.** Same attention as the first example, but approximate: tile
pruning processes the keys in blocks (tiles) and skips any tile whose best
score is already far below the running row maximum — after softmax its total
weight would be negligible, so the output is much cheaper at a bounded error.
FlashAttention is the exact special case where every tile is admitted; this
example adds the admission rule.

![An 8×8 attention matrix grouped into 2×2 tiles: near-diagonal tiles are admitted (solid indigo) and the rest are pruned (crossed out) — a pruned tile's scores and values are never read.](../assets/examples/tile-pruned-attention.svg)

$$s_{hij} = \langle k_{hj},\, q_{hi}\rangle \cdot D^{-1/2}, \qquad
\mathrm{out}_{hi} = \sum_{j} \frac{e^{s_{hij}}}{\sum_{l} e^{s_{hil}}}\, v_{hj}$$

A source tile $T$ is admitted only while
$\max_{j \in T} s_{hij} \ge m^{\mathrm{run}}_{hi} - \tau$, where
$m^{\mathrm{run}}_{hi}$ is the running row max and $\tau$ is the
`block_prune_threshold`.

How the threshold works: it is a **constructor parameter of the built-in
`gf.online_softmax()` reducer that the compiler recognizes structurally** —
not a reducer subclass mechanism. The threshold rides the reducer's
specialization key into the `gf.reducer` IR as an optional attribute, and
only the generated-CUDA dense streaming provider acts on it, emitting a
dynamic `scf.if` in the tile loop that skips a low-score tile's payload
entirely. Every other path — eager oracle, native CSR, edge-nn attention,
and all backward passes — fails closed rather than silently substituting
exact softmax, so the approximation is always an explicit choice.

Subclassing `gf.Reducer` cannot reproduce this: the structured-dense
lowering accepts only the built-in online-softmax structure, and the tile
admission logic is written into the TTIR emitter, not derived from user
algebra. Subclassing `OnlineSoftmaxReducer` and setting
`block_prune_threshold` works — but that is the same built-in parameter,
not a new extension point.

```python
--8<-- "examples/tile_pruned_attention.py:core"
```

??? example "Full source: examples/tile_pruned_attention.py (runs as-is)"

    ```python
    --8<-- "examples/tile_pruned_attention.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CUDA"

        ```text
        ### TilePrunedAttentionTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-kernel-to-ttir-dense-streaming-reduction
        passes: gf-verify-domain, gf-lower-domain-to-iter, gf-lower-iter-to-kernel, capture-structured-message, expand-reducer-regions, select-cartesian-coordinate-hierarchy, select-dense-query-neighbor-tile, lower-vector-contraction, lower-streaming-reducer-state, gf-kernel-to-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        accepted: [machine-schedule] admitted dense-query-key-tile
        unknown: [pipeline] no compiler-controlled asynchronous pipeline admitted
        remark: [planning] Cartesian relation remained implicit
        remark: [planning] multi-value message and reducer state were fused into tiles
        remark: [planning] no pairwise score or normalized-weight tensor was materialized
        executable cache: hits=0, misses=1
        ```

### Graph attention { #graph-attention }

## GAT edge attention with an nn score { #gat-edge-attention-with-an-nn-score }

**What it is.** Graph attention (GAT): instead of a dot product, each edge's
attention *score* is produced by a small neural network reading the edge
geometry and both endpoint features; the softmax over incoming edges then
weights the source features as before. The score network is where the
learnable capacity lives.

$$s_e = \mathrm{MLP}\big(\,\text{pos}_j - \text{pos}_i \,\|\, x_j \,\|\, x_i\,\big),
\qquad
\mathrm{out}_i = \sum_{e\,=\,(j \to i)}
\frac{e^{s_e}}{\sum_{e'=(j' \to i)} e^{s_{e'}}}\, x_j$$

The UDF is the first example's shape with one change — `gf.nn.trace(attn)`
produces the score. On CUDA this compiles to a row-centric fused kernel: one
program per destination row, the score MLP evaluated in registers per edge
chunk, the online-softmax state carried in loop registers — no [E, ·] score
or attention-weight tensor exists. Grad-mode calls take the fused backward
(score-chain replay + softmax Jacobian adjoint in-tile), so training
materializes nothing per edge either. The launch geometry follows the
attention defaults (`block_e=16, num_warps=1`); `gf.nn.trace` still accepts
explicit overrides.

```python
--8<-- "examples/gat_edge_attention.py:core"
```

??? example "Full source: examples/gat_edge_attention.py (runs as-is)"

    ```python
    --8<-- "examples/gat_edge_attention.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CUDA"

        ```text
        ### GATTorchExecutor
        backend: cuda
        provider: triton-nvidia@3.6.0
        lowering: gf-python-emit-edge-nn-attention-vjp
        passes: capture-edge-nn-subgraph, translate-fx-to-message-dag, prove-edge-nn-attention-structure, python-emit-edge-nn-attention-ttir, symbolic-edge-nn-attention-vjp, python-emit-edge-nn-attention-vjp-ttir, provider-compile-serialized-ttir
        provider cache key: triton-nvidia | cuda:nvidia | 3.6.0 | 2.11.0+cu128 | af81e84448f193cc | cuda:120:warp32
        remark: [planning] training path: forward persists only the O(N) softmax state (m, l); backward replays the score chain per row chunk — no [E, ·] tensor in either direction
        executable cache: hits=0, misses=2
        ```
