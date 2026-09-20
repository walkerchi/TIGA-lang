# Distributed and memory hierarchy

Multi-process graph execution and explicit memory-tier accounting. Both
examples keep the user kernel unchanged; placement and exchange are derived
below it.

- [`python examples/distributed_halo.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/distributed_halo.py)
- [`python examples/hierarchical_memory.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/hierarchical_memory.py)
- [`python examples/paged_giant_graph.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/paged_giant_graph.py)
- [`python examples/auto_offload.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/auto_offload.py)
- [`python examples/paged_cuda.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/paged_cuda.py)

The CUDA paging example persists a graph, opens it on CUDA under a 128 KiB
native-buffer budget, and checks three recurrent forward steps against NumPy.
The paging path uses `tg.Tensor`, host staging and a resident output; it does not
implement paged CUDA backward. See [the memory contract](../memory.md#large-graphs-current-boundary).

<span id="distributed-execution"></span>

## Two-process halo exchange { #two-process-halo-exchange }

**What it is.** *Sharding* splits a graph across cooperating processes so
that each process owns only a slice of the nodes and their data. When a
node's computation reads a neighbor owned by another process, that neighbor
is a *halo* (ghost) node whose value must be copied across the process
boundary — the *halo exchange*. The example also differentiates the
computation: a vector-Jacobian product (VJP) propagates gradients of a
scalar loss back to the inputs with no hand-written backward pass.
Concretely, the graph is an 8-node ring with node values `x[i] = i`; each
output sums the values of the node itself and its two ring neighbors. Rank 0
owns nodes 0–3, rank 1 owns nodes 4–7, so `out[3]` and `out[4]` each need
one value from the other rank.

![An 8-node ring sharded across two processes: rank 0 owns nodes 0–3, rank 1 owns nodes 4–7; a ghost copy of x[4] crosses the boundary so rank 0 can compute out[3], and its gradient flows back](../assets/examples/distributed-halo.svg)

$$
\mathrm{out}_i = x_{(i-1)\bmod 8} + x_i + x_{(i+1)\bmod 8},
\qquad x_i = i,
\qquad \frac{\partial \sum_i \mathrm{out}_i}{\partial x_j} = 3 .
$$

The example saves one versioned `.gfg`, reopens the same ordinary `Graph`
on two processes, reads only each rank's destination/edge pages, and runs
rank-local `Graph.halo()` MessagePassing plus the automatic VJP. The
compiler/runtime derives the forward owner→ghost exchange and the reverse
ghost-cotangent→owner accumulation below the unchanged user kernel.

```python
--8<-- "examples/distributed_halo.py:core"
```

??? example "Full source: examples/distributed_halo.py (runs as-is)"

    ```python
    --8<-- "examples/distributed_halo.py"
    ```

## Running across processes and machines { #running-across-processes-and-machines }

The example above is a complete two-process CPU forward/VJP program, not a
multi-GPU launcher. Expected rank outputs are `[8,3,6,9]` and `[12,15,18,13]`;
both source gradients are `[3,3,3,3]`.

For an executable MPI transport check, install the `mpi` and `benchmarks`
extras plus a working MPI launcher, then run the repository's
[mpi_halo_exchange.py](https://github.com/walkerchi/TIGA-lang/blob/main/benchmarks/distributed/mpi_halo_exchange.py):

```bash
mpiexec -n 2 python -m benchmarks.distributed.mpi_halo_exchange \
  --entities 64 --features 4 --repeats 3 --output output/mpi-sanity/results.json
```

Success writes `correct=true` and `gate="PASS"`. This command checks packed halo
bytes, not MessagePassing gradients. Multi-host MPI additionally needs a configured
launcher/network, matching environments and access to the same graph snapshot.

For **two machines with CUDA**, use the [complete two-host forward/VJP commands](../memory-and-distributed.md#two-host-cuda-sanity).
They start one process per host; both may use local `cuda:0`.
TCP stages data through host memory. The
[NCCL path](../memory-and-distributed.md#two-host-nccl) exchanges device buffers
through NCCL and also supports forward/VJP.

TCP `host()` and `join()` are blocking alternatives in **different processes**,
not consecutive calls in one script. Peers must be trusted (object messages use
pickle), reachable, and configured with a finite timeout that permits cold JIT.
Automatic halo scheduling does not discover hosts or launch remote processes.

<span id="memory-hierarchy"></span>

## Spilling tensors to disk { #spilling-tensors-to-disk }

`spill()` changes residency while preserving logical device and autograd history; `cpu()` is a device-copy operation. `save/load` persist value snapshots at explicit paths. Budget scope, shared views, file lifetime and CPU paging limits are explained in [memory and storage](../memory.md).

```python
--8<-- "examples/hierarchical_memory.py:core"
```

The example verifies the gradient `[0, 2, 4, 6, 8, 10, 12, 14]` and exact snapshot values. Leaving execution does not invalidate a Tensor; the example's own temporary directory cleans up its demonstration snapshot.

## A disk-resident giant graph { #a-disk-resident-giant-graph }

**What it is.** When the graph itself — not just the tensors on it — exceeds
RAM, *paging* keeps the topology (the CSR adjacency) on disk and reads it in
bounded destination-row pages. Prefetch may keep multiple pages resident and
can hide some I/O latency; it does not guarantee that computation never waits
on storage. Destination-row independence preserves per-row semantics for the
supported paged reducers and operations. Fields can join the topology on
disk: `tg.save(graph, path, fields={"src": ..., "edge": ...})` persists them
as fixed-width rows, `tg.load(path).fields("src", requires_grad=True)`
returns lazily-read shells, and both the forward pass and
`tg.autograd.grad` then run paged on supported CPU/native paths. Construction,
resident fields, prefetched pages and a materialized output can still consume
RAM; paging is not a hard process-RSS limit.

The round trip is three phases on one unchanged kernel:

$$
\text{build} \xrightarrow{\texttt{tg.save}} \texttt{.gfg on disk}
\xrightarrow[\text{stream pages}]{\texttt{tg.load} + \text{MessagePassing}}
\text{result} \xrightarrow{\texttt{out.disk(name=...)}} \text{NVMe}
\xrightarrow{\texttt{tg.from\_disk}} \text{reattach} .
$$

The example builds a 1000×1000 four-neighbor stencil (1M nodes, ~4M edges),
persists it once with `tg.save`, reopens it with `tg.load` — which reads only
the manifest, leaving the CSR on disk — runs one Jacobi smoothing step over
100k-row pages with prefetch on, spills the result under a stable name, and
reattaches it with `tg.from_disk` to verify the checksum survives the
disk → JIT → disk round trip.

Measured on a 2000×2000 stencil (4M nodes, ~16M edges, NVMe ext4, Ryzen
7 255) with `benchmarks/memory_hierarchy/paged_giant_graph.py` — identical
checksum in every row:

| page_rows | prefetch | pages | elapsed | peak RSS |
|---|---|---|---|---|
| 4,000,000 (one page) | off | 1 | 34.1 s | 3.34 GB |
| 100,000 | off | 40 | 37.7 s | 1.08 GB |
| 100,000 | on | 40 | 36.6 s | 1.09 GB |

In these archived measurements, 100,000-row pages reduce peak RSS from 3.34 GB
to about 1.09 GB, with a longer completed-call time. The elapsed time includes
page loading and computation. Use the benchmark with the intended hardware and
page size to evaluate this tradeoff.

```python
--8<-- "examples/paged_giant_graph.py:core"
```

??? example "Full source: examples/paged_giant_graph.py (runs as-is)"

    ```python
    --8<-- "examples/paged_giant_graph.py"
    ```

## Automatic offload under a RAM budget { #automatic-graph-offload }

**What it is.** The explicit `tg.save`/`tg.load` dance above exists for
cross-process handoff. A per-CSR threshold can make the paging decision automatically: while
`tg.runtime.auto_offload(ram=...)` is active, every CSR builder
(`Graph.from_csr`, `Graph.stencil`, `Graph.cat`, ...) checks the CSR byte
size and, over budget, persists the topology and returns a paged graph —
the unchanged native kernel call then streams pages with prefetch.
The threshold is checked **after CSR construction**; it is not an out-of-core
builder or a total-RSS/working-set budget:


```python
--8<-- "examples/auto_offload.py:core"
```

??? example "Full source: examples/auto_offload.py (runs as-is)"

    ```python
    --8<-- "examples/auto_offload.py"
    ```

The pipeline depth — pages read ahead concurrently — defaults to 2 and is
tunable per call (`prefetch_depth=4`) or process-wide
(`TIGA_PAGED_PREFETCH_DEPTH`); page size follows `page_rows=` or
`TIGA_PAGED_PAGE_ROWS`. Offloaded graphs land in a per-process
directory cleaned up at exit, or under `TIGA_SPILL_DIR` when the
graph should outlive the process. This path requires CPU/native graphs and supported paged operations;
fields can be offloaded too via `tg.save(..., fields=...)`, and both forward
and backward run paged.
??? info "Measured compilation artifacts (Ryzen 7 255 · RTX 5070 Ti)"

    === "This machine"

        ```text
        ### Smoothing
        backend: cpu:0
        provider: tiga-runtime
        lowering: gf-tensor-relation-autograd
        passes: bind-static-csr-snapshot, capture-message-passing-udf, analyze-reducer-algebra, lower-csr-to-gather-segment, fuse-edge-node-regions
        remark: [planning] edge/node UDFs remain in the differentiable Tensor DAG
        remark: [planning] gf-tensor-vjp generates CSR gather/segment-sum adjoints
        remark: [planning] reducer lowering: builtin-additive-state
        variant cache: hits=0, misses=10
        ```
