# Reducers

Prerequisite: [Programming model](programming-model.md).
A [reducer](https://en.wikipedia.org/wiki/Fold_(higher-order_function))
specifies how messages arriving at one destination combine. Start with a
built-in; the custom algebra below is needed only when built-ins cannot
express the computation.

## Choose a built-in reducer { #choose-reducer }

| Desired result | Choice | Important behavior |
|---|---|---|
| Sum of messages | `tg.sum()` | An empty row produces 0 by default |
| Mean of messages | `tg.mean()` | An empty row produces NaN, not automatically 0 |
| Product of messages | `tg.prod()` | An empty row produces 1 |
| Score-weighted mean | `tg.online_softmax()` | Pass score and value inside `edge()`; execution coverage has additional constraints |

For example, select mean in an ordinary MessagePassing definition:

```python
import tiga as tg

class NeighborMean(tg.MessagePassing):
    reducer = tg.mean()

    def edge(self, src, dst, edge):
        return src.x
```

Graph and field bindings are unchanged. The
[neighbor-mean example](examples/message-passing.md#user-defined-reducer)
shows a complete call; the [API reference](api.md#reducers) lists built-in signatures.
Graph, dtype and execution path must still meet the [support matrix](roadmap.md#feature-support).

## Advanced: reducer representation { #reducer-representation }

The following sections explain custom reducers, algebra contracts and compilation.
They are not prerequisites for using a built-in.

A reducer is **executable algebra**, not a name like `"sum"`. It is the third
member of every MessagePassing program: the edge UDF produces messages, the
reducer defines how they combine into one value per destination node. The four
methods below are captured as first-class `gf.reducer` IR and fused with the
relation traversal before provider lowering — which is why the compiler can
generate the VJP of a hand-written reduction.

For one destination node `i` with incoming edge messages `m_e`, the semantics
are a fold:

$$
\text{out}_i \;=\; \text{finalize}\!\Big( \bigoplus_{e\,=\,(j \to i)} \text{lift}(m_e) \Big),
\qquad
s \oplus s' = \text{combine}(s, s')
$$

The same fold as an ordinary Python loop — where each method runs:

```python
for i in range(num_dst):                 # per destination node
    state = identity()                   # before its edges
    for e in edges_into(i):
        m = edge(src, dst, edge)         # the edge() UDF produces the message
        state = combine(state, lift(m))
    out[i] = finalize(state)             # after its edges
```

This shows the logical order, a left fold. A declared associative/commutative
algebra frees the compiler to combine in any order or as a balanced tree.

![Reducer anatomy: messages are lifted into state, merged pairwise by combine, and mapped to the result by finalize](assets/reducer-anatomy.svg)

## Usage patterns

Every supported way to drive a reducer, each with a runnable example:

**`tg.sum(identity=…)`.** The additive monoid, lowered to a direct segment
sum; [gcn.py](examples/message-passing.md#gcn-aggregation).

$$
\text{out}_i = \sum_e m_e
$$

**`tg.mean()`.** The neighbor mean as built-in tuple state
`(sum, count)`; [inline below](#gfmean).

$$
\text{out}_i = \frac{1}{\deg i} \sum_e m_e
$$

**`tg.prod()`.** The product monoid, promoted to a zero-safe product IR;
[inline below](#gfprod).

$$
\text{out}_i = \prod_e m_e
$$

**`tg.online_softmax()`.** Flash attention as a monoid over tuple state
`(m, l, a)`, called as `self.reducer(score, value)`;
[tile_pruned_attention.py](examples/attention.md#tile-pruned-sparse-attention).

**Tile pruning.** `block_prune_threshold` lets the dense CUDA path skip
sub-threshold score tiles (a semantic approximation, off by default);
[tile_pruned_attention.py](examples/attention.md#tile-pruned-sparse-attention).

**Tuple state for a mean, written by hand.** The algebra behind the
`tg.mean()` built-in spelled out as a custom reducer;
[custom_reducer.py](examples/message-passing.md#user-defined-reducer).

**One-pass moments.** Variance via `(count, sum, sum_of_squares)`, proven
componentwise additive with a generated VJP;
[inline below](#neighbor-variance-in-one-pass).

**Product monoid, written by hand.** The algebra behind the `tg.prod()`
built-in spelled out for gate reliability;
[inline below](#product-of-edge-gates).

$$
\prod_e p_e
$$

**Hand-written online softmax.** The built-in algebra spelled out as a
user reducer, taking the same lowering family;
[inline below](#online-softmax-written-by-hand).

**Multi-message reducer calls.** `self.reducer(score, value)` stages a
two-field binding instead of a value;
[full_attention.py](examples/attention.md#full-attention-no-mask).

**torch.nn-produced messages with `tg.sum()`.** The message is a fused,
trainable edge MLP; [edge_nn_message_passing.py](examples/programs-and-interop.md#edge-nn-modules-with-a-fused-tile-kernel).

**torch.nn-produced scores with `tg.online_softmax()`.** GAT-style
attention where the edge *score* is a small network; compiles to a
row-centric fused kernel (forward and backward);
[gat_edge_attention.py](examples/attention.md#gat-edge-attention-with-an-nn-score).

**Reducer inspection.** `reducer.mlir(...)` returns the captured
`gf.reducer` op as text; [custom_reducer.py](examples/message-passing.md#user-defined-reducer).

## The four region methods

| Method | Role | Default |
|---|---|---|
| `identity(self)` | state of an empty (degree-0) row | *required* |
| `lift(self, *messages)` | map one edge's message(s) to state | single message passthrough, or tuple |
| `combine(self, left, right)` | merge two states | *required* |
| `finalize(self, state)` | map the row's state to the node result | identity |

State may be a scalar or a tuple. A neighbor mean lifts values into
`(value, count)`, combines the two components by addition, and finalizes by
dividing the sum by the count. See the runnable
[custom reducer example](examples/message-passing.md#user-defined-reducer).

## Class attributes and `__init__` { #class-attributes-and-init }

| Attribute | Meaning | Effect on compilation |
|---|---|---|
| `name` | IR symbol name (`"custom"` default) | part of the cache/specialization key |
| `associative` | grouping does not change the algebraic result | **required `True` for native parallel lowering**; never proven from Python source — the declaration is the contract |
| `commutative` | swapping operands does not change the algebraic result | allows the combine tree to reorder operands |
| `deterministic` | instance flag, `__init__(*, deterministic=False)` | `True` picks a fixed-order sequential reduction (bitwise-stable); `False` allows a balanced tree |

These flags are trusted, not inferred: if `associative = True` is declared for
an algebra that is not associative, the generated parallel code is wrong and
that is a user error.

## Calling a reducer inside `edge`

An ordinary single-message `edge()` returns `src.value` directly; the class
attribute already declares the reducer. In particular, built-in `tg.sum()`
expects a Tensor directly, not `self.reducer(...)`. The following is the
explicit binding form for a custom reducer; the multi-input online-softmax
interface uses a similar form. It packages inputs for the declared reducer,
returning a **staged binding**, not performing aggregation inside one edge:

```python
def edge(self, src, dst, edge):
    return self.reducer(src.value)          # ReducerCall(reducer, (value,))
```

- At least one message is required (`TypeError` otherwise).
- Multiple messages are packaged as inputs; `lift` then constructs internal
  state. A message tuple is not itself the reducer state:
  `return self.reducer(score, payload)`.
- The staged item is what lets the compiler fuse lift into the edge region and
  keep combine/finalize in the reduction.

## Built-in reducers, op by op

### `tg.sum(identity=0)` { #gfsumidentity0 }

The additive monoid — every method is the trivial one:

$$
\text{identity}() = 0, \qquad
\text{lift}(m) = m, \qquad
a \oplus b = a + b, \qquad
\text{finalize}(s) = s
$$

The result is the sum of incoming messages, lowered directly to a segment sum.
`identity` changes the value a degree-0 row produces.

### `tg.mean()` { #gfmean }

The neighbor mean as built-in tuple state `(s, n)`:

$$
\begin{aligned}
\text{identity}() &= (0, 0), \qquad
\text{lift}(m) = (m, 1) \\
(s, n) \oplus (s', n') &= (s + s',\; n + n') \\
\text{finalize}\big((s, n)\big) &= s / n
\end{aligned}
$$

The result is the mean of incoming messages. The algebra is proven
componentwise additive, so the native path lowers it to two segment sums plus
a final divide, with a generated VJP. A degree-0 row finalizes `0/0` and
produces NaN — if isolated nodes should yield zeros, use `tg.sum()` and
divide by a clamped degree.

### `tg.prod()` { #gfprod }

The multiplicative monoid:

$$
\text{identity}() = 1, \qquad
\text{lift}(m) = m, \qquad
a \oplus b = a \cdot b, \qquad
\text{finalize}(s) = s
$$

The result is the product of incoming messages, e.g. gate reliability: the
probability that every gate into `i` stays open. The compiler promotes it to
a zero-safe product IR with a generated VJP; a degree-0 row yields the
identity `1`.

### `tg.online_softmax()` { #gfonline_softmax }

A softmax-weighted mean streamed over tuple state `(m, l, a)` — running
maximum, normalizer, weighted sum:

$$
\begin{aligned}
\text{identity}() &= (-\infty,\; 0,\; 0) \\
\text{lift}(s, v) &= (s,\; 1,\; v) \\
(m_1, l_1, a_1) \oplus (m_2, l_2, a_2) &=
\big(m,\;\; l_1 e^{m_1 - m} + l_2 e^{m_2 - m},\;\; a_1 e^{m_1 - m} + a_2 e^{m_2 - m}\big) \\
\text{finalize}\big((m, l, a)\big) &= a \,/\, l
\end{aligned}
$$

Here `m = max(m1, m2)` rescales both partial states into a common range
before merging; the result is a softmax-weighted mean of the values. Rescaling lives
inside `combine`, so partial states merge in any order and no score matrix is
ever materialized — this is flash attention's online softmax, written as a
monoid. It is called in `edge` as `self.reducer(score, value)`. The score may
itself be a `tg.nn.trace`'d network (GAT-style learned attention): on CUDA
that compiles to a row-centric fused kernel with a fused backward — see the
[GAT example](examples/attention.md#gat-edge-attention-with-an-nn-score).

- `accumulation_dtype` — dtype of the streaming state.
- `block_prune_threshold` in `(0, 1]` — a **semantic approximation
  policy**, not a schedule hint: a dense streaming implementation may skip a
  whole score tile when its maximum softmax weight falls below this ratio of
  the running block maximum. `None` (default) keeps exact online softmax.
  Currently only honored on the generated-CUDA DenseGraph path.

**Why no `max` / `min` / `median` built-in?** Max has a simple algebra, but
the native differentiable vocabulary has no `maximum` op with a subgradient
VJP yet. `tg.online_softmax()` computes a smooth weighted mean, not an exact
max, and cannot silently replace it. Median and quantiles are
different in kind: they are **not associative monoids** — no fixed-size state
`combine(s, s')` can accumulate a median — so they are fundamentally outside
one-pass reducer semantics and need a two-pass algorithm or a sketch
(t-digest, moments).

## Neural networks in the message or the score

A `tg.nn.trace`'d `torch.nn` module fuses into the reduction at two points —
both compile to single fused CUDA kernels in which per-edge activations never
leave the tile, and both are trainable end to end (fields, positions and all
module weights receive generated gradients):

**The nn produces the message**, combined by `tg.sum()` — compiled to an
edge-centric tile kernel
([example](examples/programs-and-interop.md#edge-nn-modules-with-a-fused-tile-kernel)):

$$
m_e = \operatorname{MLP}_\theta\big([\,\mathbf{x}_j \,\|\, \mathbf{x}_i\,]\big),
\qquad
\text{out}_i = \sum_{e=(j \to i)} m_e
$$

**The nn produces the score**, combined by `tg.online_softmax()` — GAT-style
learned attention, compiled to a row-centric online-softmax tile kernel with
a fused backward
([example](examples/attention.md#gat-edge-attention-with-an-nn-score)):

$$
s_e = \operatorname{MLP}_\theta\big([\,\mathbf{h}_i \,\|\, \mathbf{h}_j\,]\big),
\qquad
\text{out}_i = \sum_{j} \operatorname{softmax}_j(s_{ji})\, \mathbf{v}_j
$$

The score chain must produce one scalar per edge (`out_features=1`), and the
softmax `value` must be a field gather. An nn inside a *custom* reducer
matches no fusion form and falls back to the exact eager oracle — correct,
not fast. The op whitelist, LayerNorm support and launch-geometry knobs live
in the [message passing guide](message-passing.md#edge-nn-modules-cuda-torch-interop).

## Inventing a reducer { #inventing-your-own }

Three algebras that actually run on the native path — each one is recognized
by the compiler as a proven form, so the forward kernel and its VJP are both
generated. Built-ins are conveniences, not privileged compiler paths.

### Neighbor variance in one pass

One traversal accumulates `(count, sum, sum_of_squares)` for feature
normalization layers that need neighborhood statistics. Finalization computes:

$$
\text{out}_i = \mathbb{E}[x^2] - \mathbb{E}[x]^2
$$

```python
class Variance(tg.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 0.0, 0.0, 0.0                 # (count, sum, sum of squares)

    def lift(self, value):
        return 1.0, value, value * value

    def combine(self, left, right):
        return (left[0] + right[0],
                left[1] + right[1],
                left[2] + right[2])

    def finalize(self, state):
        mean = state[1] / state[0]
        return state[2] / state[0] - mean * mean


class NeighborVariance(tg.MessagePassing):
    reducer = Variance()

    def edge(self, src, dst, edge):
        return self.reducer(src.x)


out = NeighborVariance()(graph=graph, src={"x": x}, dst={})  # (N,)
```

Structurally proven **componentwise additive**. Note that `finalize` also
runs on `identity()` for degree-0 rows — here an isolated node divides
`0/0`, so pick an identity/finalize pair with empty rows in mind.

### Product of edge gates

Reliability-style aggregation multiplies incoming edge probabilities: the probability
that *every* gate into `i` stays open. This is exactly the algebra the
`tg.prod()` built-in ships — spelled out here to show the mechanics:

```python
class Product(tg.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return 1.0

    def combine(self, left, right):
        return left * right


class GateReliability(tg.MessagePassing):
    reducer = Product()

    def edge(self, src, dst, edge):
        return self.reducer(edge.open_probability)
```

`lift` and `finalize` inherit the base-class defaults (passthrough). The
compiler proves the **product monoid** from the expression tree — identity
`1`, passthrough lift, `mul` combine — and promotes it to a zero-safe product
IR with a generated VJP.

### Online softmax, written by hand

`tg.online_softmax()` is not magic — the same algebra as a user reducer is
structurally proven **stable-weighted** and takes the same lowering family:

```python
class Softmax(tg.Reducer):
    associative = True
    commutative = True

    def identity(self):
        return float("-inf"), 0.0, 0.0       # (max, denominator, numerator)

    def lift(self, score, value):
        return score, 1.0, value

    def combine(self, left, right):
        m = left[0].maximum(right[0])
        l = left[1] * (left[0] - m).exp() + right[1] * (right[0] - m).exp()
        a = left[2] * (left[0] - m).exp() + right[2] * (right[0] - m).exp()
        return m, l, a

    def finalize(self, state):
        return state[2] / state[1]
```

The captured scalar vocabulary is arithmetic, `.exp()` and `.maximum()` over
tuple state. Algebras matching no proven form still run — as a general
explicit reduction tree (arithmetic and `exp`), correctness-only, never a
performance claim.

## How the compiler picks a lowering

On the native path, reducers are matched by structure, in this order:

1. `tg.sum()` → direct segment sum; an nn-subgraph message lowers further to
   the fused edge-centric tile kernel (and its symbolic-VJP backward);
2. `tg.mean()` → proven componentwise-additive tuple state: two segment sums
   plus a final divide, with a generated VJP;
3. `tg.prod()` → proven product monoid, promoted to a zero-safe product IR
   with a generated VJP;
4. `tg.online_softmax()` → tiled stable streaming path; an nn-subgraph score
   lowers further to the row-centric fused attention kernel (and its fused
   backward);
5. user reducers proven **stable-weighted**, **product monoid**
   (promoted to zero-safe product IR), or **componentwise additive** (tuple
   state) → generated reduction with generated VJP;
6. anything else → a general explicit reduction tree: sequential when
   `deterministic=True`, balanced otherwise; scalar result.

Shapes that match no proven form remain correctness-only claims, never
performance claims.

## Inspecting a reducer

`reducer.mlir(message_dtypes=(tg.float32,), symbol="user_mean")` captures the
four regions as a verified `gf.reducer` op and returns its text — the same
artifact the custom reducer example points at. A reducer's flags and
constructor configuration form its specialization key, so two reducers with
different algebra never share a compiled cache entry.

See the [API reference](api.md#reducers) for the flat listing.
