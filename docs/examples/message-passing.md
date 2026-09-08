# Message passing and graph algorithms

Static-topology MessagePassing programs and a fixed-iteration graph algorithm
probe. The compiler generates the relation traversal and its VJP; the user
writes only the UDFs. The [message passing guide](../message-passing.md)
explains the interface. Each section shows the minimal core snippet; the
complete runnable program is one click away in the collapsed source block.

- [`python examples/message_passing_autograd.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/message_passing_autograd.py)
- [`python examples/gcn.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/gcn.py)
- [`python examples/diffusion.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/diffusion.py)
- [`python examples/custom_reducer.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/custom_reducer.py)
- [`python examples/compiler_probes/pagerank.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/compiler_probes/pagerank.py)

### Basic aggregation { #basic-aggregation }

## GCN aggregation { #gcn-aggregation }

**What it is.** A graph convolutional network (GCN) layer: every node of a
graph carries a feature vector (here F = 2 numbers), and the layer replaces
each node's features with the sum of the feature vectors of the nodes that
link to it, each scaled by one scalar weight per edge. In linear-algebra
terms this is a weighted sparse matrix–matrix multiplication (SpMM): the
weighted adjacency matrix times the feature matrix. The example also
differentiates the sum of all outputs back onto the features and the weights.

![GCN aggregation data flow: node feature rows and per-edge weights are combined per edge by the user UDF, then summed per destination into the output feature matrix](../assets/examples/gcn.svg)

$$
\text{out}_{i,f} \;=\; \sum_{e\,=\,(j \to i)} w_e \cdot x_{j,f}
$$

Native multi-feature aggregation with edge broadcasting and automatic
gradients, matching the GCN tensor shape. The GCN layer is user code, not a
library operator.

```python
--8<-- "examples/gcn.py:core"
```

??? example "Full source: examples/gcn.py (runs as-is)"

    ```python
    --8<-- "examples/gcn.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CPU"

        ```text
        ### WeightedFeatureAggregation
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

    === "CUDA"

        ```text
        ### WeightedFeatureAggregation
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

## Directional diffusion { #directional-diffusion }

**What it is.** Diffusion: a quantity stored on the nodes of a graph (heat,
concentration) flows along the edges from larger values toward smaller ones
until it evens out. The flux into a node is the sum over its incoming edges
of the edge's conductivity times the difference between the neighbor's value
and the node's own. The example advances one explicit Euler step — each node
moves by a small time step Δt times its net flux — on a 4-node ring where
every node exchanges with two neighbors.

![Graph diffusion: a four-node ring of values; every edge carries conductivity times the endpoint difference toward its destination, and each node steps by dt times its net inflow](../assets/examples/diffusion.svg)

$$
\text{flux}_i \;=\; \sum_{e\,=\,(j \to i)} c_e \, (u_j - u_i),
\qquad
u^{+}_i \;=\; u_i + \Delta t \cdot \text{flux}_i
$$

Both the field `u` and the edge conductivities receive compiler-generated
gradients.

```python
--8<-- "examples/diffusion.py:core"
```

??? example "Full source: examples/diffusion.py (runs as-is)"

    ```python
    --8<-- "examples/diffusion.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CPU"

        ```text
        ### Diffusion
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

    === "CUDA"

        ```text
        ### Diffusion
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

### Custom UDFs and reducers { #custom-udfs-and-reducers }

## Differentiable MessagePassing UDF { #differentiable-messagepassing-udf }

**What it is.** This example treats nodes as points carrying a temperature
and a bias, and edges as conductors: each edge forwards its conductivity
times the temperature of its source node to its destination, which adds its
own bias. The example then asks for the derivative of the total output with
respect to every input, delivered as a vector–Jacobian product (VJP) — the
adjoint of the forward computation — without any hand-written backward code.

The example computes a conductive aggregation with a destination bias,

$$
\text{out}_i \;=\; b_i \;+\; \sum_{e\,=\,(j \to i)} c_e \cdot T_j
$$

and then differentiates `sum(out)` with respect to all three inputs. The
compiler generates the relation gather/segment primitives and the VJP from
the same IR; only the UDFs are user code. The example also shows the explicit
`save` checkpoint; on CUDA `auto` chooses between saved relation gathers and
recomputation.

```python
--8<-- "examples/message_passing_autograd.py:core"
```

??? example "Full source: examples/message_passing_autograd.py (runs as-is)"

    ```python
    --8<-- "examples/message_passing_autograd.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CPU"

        ```text
        ### ConductiveAggregation
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

    === "CUDA"

        ```text
        ### ConductiveAggregation
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

## User-defined reducer { #user-defined-reducer }

**What it is.** How to compute a mean when the only thing the machine can do
in parallel is combine values pairwise. The trick is to carry a pair —
running sum and item count — instead of a single number: each value v becomes
(v, 1), pairs merge by adding both components, and a final division turns the
accumulated pair into the mean. Because merging is associative and
commutative, the combine order — and hence the parallel schedule — is free.

![Reducer algebra for a mean: lift maps a value to a (sum, count) pair, combine adds two pairs componentwise, finalize divides; a concrete example combines (1.0, 1) and (4.0, 1) into (5.0, 2) and then 2.5](../assets/examples/custom-reducer.svg)

$$
\text{lift}(v) = (v,\, 1)
\qquad
(s_l, n_l) \oplus (s_r, n_r) = (s_l + s_r,\; n_l + n_r)
\qquad
\text{mean}_i = \frac{s_i}{n_i}
$$

Where each method runs — the same computation as an ordinary Python loop:

```python
for i in range(num_dst):                 # per destination node
    state = identity()                   # (0.0, 0.0) — before its edges
    for e in edges_into(i):
        message = edge(src, dst, edge)   # your edge() UDF, here src.value
        state = combine(state, lift(message))
    out[i] = finalize(state)             # s / n — after its edges

# node 0 traces:  (0.0, 0.0) → combine((0.0,0.0), lift(1.0)) = (1.0, 1)
#                          → combine((1.0,1),   lift(4.0))   = (5.0, 2)
#                          → finalize((5.0, 2)) = 2.5
```

The loop above shows the logical order, a left fold. Because the algebra is
declared associative and commutative, the compiler may combine in any order
or as a tree — that is what makes the parallel schedule free. It proves the
tuple state componentwise additive, lowers the four regions, and generates
the backward through the mean as well.

```python
--8<-- "examples/custom_reducer.py:core"
```

??? example "Full source: examples/custom_reducer.py (runs as-is)"

    ```python
    --8<-- "examples/custom_reducer.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CPU"

        ```text
        ### NeighborMean
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: proved-componentwise-additive-udf
        executable cache: hits=0, misses=1
        ```

    === "CUDA"

        ```text
        ### NeighborMean
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: proved-componentwise-additive-udf
        executable cache: hits=0, misses=1
        ```

### Iterative control flow { #iterative-control-flow }

## Fixed-iteration PageRank probe { #fixed-iteration-pagerank-probe }

**What it is.** PageRank: the algorithm that ranks web pages by importance.
Every page starts with equal rank; in each round a page divides its rank
equally among its outgoing links and collects the rank arriving through its
in-links, blended with a small constant (1 − damping)/N that models jumping
to a random page. A page with no outgoing links (a dangling node) would leak
rank out of the system, so its rank is redistributed uniformly to all pages.
Here: 4 pages, 3 links, damping 0.85, 20 rounds.

![PageRank probe: four pages with three links forming a cycle among pages 0, 1 and 2 plus a dangling page 3; twenty rounds are captured once as a single gf_control.repeat op producing the final rank vector](../assets/examples/pagerank.svg)

$$
\text{base} = \frac{1-d}{N} + \frac{d}{N} \sum_{i\ \mathrm{dangling}} \text{rank}_i,
\qquad
\text{rank}'_i = \text{base} + d \sum_{e\,=\,(j \to i)} \frac{\text{rank}_j}{\mathrm{deg}^{\text{out}}_j}
$$

The iteration is a plain Python `for` loop under `@gf.jit`; 20 iterations are
captured once as a single `gf_control.repeat` op — the loop stays rolled, no
unrolling (the tests verify this IR shape and the rank values against a
framework-independent reference). It is a compiler probe, not a
workload-specific core operator or a performance claim.

```python
--8<-- "examples/compiler_probes/pagerank.py:core"
```

??? example "Full source: examples/compiler_probes/pagerank.py (runs as-is)"

    ```python
    --8<-- "examples/compiler_probes/pagerank.py"
    ```

??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "CPU"

        ```text
        ### PageRankStep
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```

    === "CUDA"

        ```text
        ### PageRankStep
        backend: cuda:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        executable cache: hits=0, misses=1
        ```
