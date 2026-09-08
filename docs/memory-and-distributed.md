# Memory hierarchy and distributed execution

These are compiler dimensions, not alternative user APIs. The same
`MessagePassing` program must remain valid whether a relation is resident in
one GPU, streamed from NVMe, or partitioned across ranks.

!!! note "Current status"

    Single-node capacity-accounted RAM/pinned/HBM/NVMe execution is implemented,
    including compiler-planned transfer/release and native async DMA. Typed
    owned/ghost/halo tasks, paged `.gfg` shards and a real two-process MPI
    forward/VJP path are executable. CPU communication/interior overlap has a
    controlled-link benchmark; CUDA has native buffer binding and verified
    stream dependency ordering. A rank-one NCCL test proves provider binding,
    not peer communication. Real 2+ GPU NCCL/RCCL correctness, profiler overlap
    and performance remain open gates.

## Memory hierarchy

Tiga separates a **logical value** from its **physical instances**. A
field named `state` at version 7 may simultaneously exist as an HBM buffer, a
pinned staging buffer and an NVMe spill page; the runtime tracks which
instances are live and which one is newest.

![Memory hierarchy ladder: register, shared, device, host-pinned, RAM, NVMe and remote tiers with their managers and transfer mechanisms](assets/memory-hierarchy-ladder.svg)

The ladder has three management regimes:

- **kernel-managed** (`register`, `shared`) — never allocatable; the generated
  kernel code owns them;
- **runtime-allocatable** (`device`, `host-pinned`, `ram`, `nvme`) — created
  through `HierarchyRuntime.allocate`, transferred through
  `HierarchyRuntime.transfer`;
- **transport-managed** (`remote`) — another rank's shard, reached only through
  halo exchange (next section).

### Capacity and version accounting

Every instance is created with an explicit byte capacity, and each tier has a
budget. The accounting rule is a hard gate, not a hint:

$$
\text{live}(t) \;=\; \sum_{i\ \text{live on}\ t} \text{capacity}_i \;\le\; \text{budget}(t)
$$

Overflow raises `MemoryError` — **spilling is an explicit transfer to another
tier, never a silent runtime decision**. Two contracts keep instances coherent:

- a `transfer(source, destination)` requires the same logical name, the same
  `version` and the same byte size, and waits on both sides' pending work
  (WAW ordering) before starting;
- `runtime.latest("state")` resolves the live instance with the highest
  version, so a compiler plan never has to guess which copy is current.

All transfers are asynchronous completions (`transfer(...)` returns a handle
with `.ready` / `.wait()`): buffer copies ride a CUDA or CPU `Stream` with
pooled events, and NVMe pages ride the runtime's file-I/O thread pool.

Both the user-facing `.disk()` spill and compiler bundle plans bottom out in
this runtime. One manual RAM → NVMe spill, written against it directly:

```python
nbytes, version = 256 << 20, 7        # a 256 MiB logical tensor "state" at version 7

with gf.runtime.HierarchyRuntime(budgets={"ram": 1 << 30, "nvme": 4 << 30}) as rt:
    hot = rt.allocate("state", version, tier="ram",  capacity_bytes=nbytes)
    cold = rt.allocate("state", version, tier="nvme", capacity_bytes=nbytes)
    rt.transfer(hot, cold).wait()       # async spill; same name + version + size required
    hot.close()                         # live("ram") drops; latest("state") is now cold
```

The runnable version — including the round-trip integrity check — is
[examples/hierarchical_memory.py](examples/distributed-memory.md#spilling-tensors-to-disk);
pinned↔HBM and RAM↔NVMe bandwidth is measured by
`python -m benchmarks.memory_hierarchy.transfer --quick`.

### Compiler bundle plans

Above this runtime sits the compiler's physical plan: an
`ExecutableBundlePlan` is a validated DAG of invocations plus
`ResourceRequirement`s (name, memory space, capacity, snapshot version).
`allocate_hierarchy(runtime)` maps requirements onto tier instances, and three
reserved symbols give the plan explicit storage control: `__gf_transfer`
(a tier move), `__gf_release` (an instance close) and `__gf_join`.

The legality rule is worth stating plainly: **every write/write or read/write
conflict must be ordered by an explicit dependency** unless two accesses are
proven disjoint through the same partitioning — the runtime never invents a
dependency. A bundle that violates this is rejected at construction, not
debugged after a corrupt run.

## Distributed execution

A distributed graph is still an ordinary `Graph`. `graph.halo(mesh, ...)`
attaches declarative placement — no data moves, no subtype appears:

```python
mesh = gf.DeviceMesh("cuda", (2, 4), names=("rack", "gpu"))
graph = graph.halo(mesh, partition=gf.ByDestination(mesh_axis="gpu"), depth="auto")
```

### Ownership and ghosts, exactly

With $N$ destination entities and $W$ ranks, the default balanced partition
gives rank $r$ the contiguous row range

$$
\text{owned}_r \;=\; \big[\, r\big\lfloor \tfrac{N}{W} \big\rfloor + \min(r,\; N \bmod W),
\;\; (r{+}1)\big\lfloor \tfrac{N}{W} \big\rfloor + \min(r{+}1,\; N \bmod W) \big)
$$

and the ghost set is derived from the relation, not declared:

$$
H_r \;=\; \{\, j \;\mid\; \exists\, e = (j \to i),\; i \in \text{owned}_r,\; j \notin \text{owned}_r \,\}
$$

`derive_halo_map` computes this per rank from owned CSR rows alone; for a paged
`.gfg` store each rank reads only its own row range, so no rank ever
materializes the global adjacency.

![Halo exchange: an 8-node ring on two ranks; ghosts are read at the boundary, values flow owner to ghost in the forward pass and cotangents accumulate back in the VJP; interior compute overlaps the exchange](assets/halo-exchange.svg)

### Forward: interior/boundary split and overlap

Before execution the compiler splits owned rows into **interior** (references
only owned sources — no halo dependency) and **boundary** (may reference
ghosts). The schedule is `interior ‖ halo → boundary`:

- the halo exchange (`pack_halo` → `exchange_packed` → `unpack_halo`, values
  flowing owner → ghost) starts first;
- interior compute launches immediately, in parallel with the exchange;
- boundary compute waits for the ghosts.

$$
T_{\text{serial}} = t_{\text{comm}} + t_{\text{int}} + t_{\text{bnd}}
\qquad\Longrightarrow\qquad
T_{\text{overlap}} = \max(t_{\text{comm}},\, t_{\text{int}}) + t_{\text{bnd}}
$$

Every execution records a typed trace
(`DistributedRuntime.last_execution_trace`, schema
`tiga.distributed-execution-trace.v1`) so overlap claims come from
measured schedules, not hopes. The controlled-link CPU measurement is on the
[benchmark results](benchmark-results.md) page (Distributed tab).

### Backward: the adjoint is also a halo exchange

The forward pass wraps each combined source field as a
`distributed_halo_snapshot` leaf, so the generated VJP knows the reverse
communication pattern without any user code: ghost cotangents are sent back to
their owners and **accumulated** into the owned rows —

$$
\bar{x}_j \;\mathrel{+}=\!\! \sum_{\substack{e = (j \to i) \\ i\ \text{owned by } r}} \!\! \bar{m}_e^{(r)}
\quad \text{for every remote rank } r
$$

which is exactly the adjoint of the forward owner→ghost gather. The
[distributed halo example](examples/distributed-memory.md#two-process-halo-exchange)
verifies this end to end on an 8-node ring: each rank's local gradient is
uniformly 3, including the contributions that cross the rank boundary.

### Runtimes and transports

Execution requires an active runtime; calling a distributed graph without one
fails closed (`NotImplementedError`):

```python
with gf.distributed.DistributedRuntime(transport):
    out = Diffusion()(graph=graph, src={"u": local_u}, dst={"u": local_u})
```

Transports are deployment plugins selected at process initialization — kernel
and graph code never name one:

- `PipeTransport` — stdlib `multiprocessing` pipes (used by the example and
  the two-process tests);
- `TCPTransport` / `from_provider("tcp")` — dependency-free cross-machine
  transport over plain sockets (rendezvous, rank handshake, framed streams);
- `DistributedRuntime.from_provider("mpi")` — a thin mpi4py wrapper;
- `from_provider("nccl", rank=..., world_size=..., communicator_id=..., ...)`
  binds a real `libnccl` communicator (grouped send/recv only; rank 0 creates
  the 128-byte `ncclUniqueId`, the launcher broadcasts it);
- third parties register through the `tiga.transport` entry-point group.

Persistent topology keeps the same API: `gf.save(graph, "mesh.gfg")` writes a
versioned paged store (manifest + `row_ptr.bin` + `col_idx.bin`), and
`gf.load("mesh.gfg").halo(mesh, depth="auto")` returns to the same partitioned
snapshot with each rank reading only its own pages. `gf.load` reads the
manifest only; `resolve_csr()` remains an explicit debug materialization.

Measured transfer, halo-exchange, overlap and NCCL evidence for these paths is
collected on the [benchmark results](benchmark-results.md) page.
