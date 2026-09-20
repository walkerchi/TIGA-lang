# Memory hierarchy and distributed execution

Ordinary programs use [memory and storage](memory.md). This page describes the compiler-facing [memory hierarchy](https://en.wikipedia.org/wiki/Memory_hierarchy) and distributed execution mechanisms.

!!! note "Current boundary"

    Native-Tensor [LRU eviction](memory.md#automatic-eviction), compiler physical instances and graph paging have separate ownership and budgets. CUDA paging supports FP32 forward with host staging and a resident output; backward is CPU-only. Distributed TCP/NCCL execution supports forward/VJP and completes communication before computation. See the [support matrix](roadmap.md#feature-support) for coverage and [performance measurements](experiments.md) for workload results.

## Memory hierarchy

A logical value may have several explicitly managed physical instances. `HierarchyRuntime` tracks their logical name, version, capacity and completion; it does not automatically turn every public Tensor into such an instance.

![Memory tiers and their managers](assets/memory-hierarchy-ladder.svg)

Register/shared storage belongs to generated kernels. `device`, `host-pinned`, `ram` and `nvme` are runtime-allocatable. `remote` denotes transport-managed data, not a locally allocatable buffer.

### Capacity and version accounting

An instance allocation checks its runtime's per-tier budget. Native buffers and temporary NVMe instances also participate in an active `tg.execution` budget; these are distinct checks, not twice the allocation. Capacity counts live allocations, not process RSS.

A transfer requires matching logical name, version and capacity. Its completion is available through `.ready` / `.wait()`. Closing a source waits for outstanding readers before releasing its bytes; writing a destination also waits for its previous readers. `latest(name)` selects the highest live version; equal-version copies have no implied preferred tier.

This complete CPU example moves eight bytes through an NVMe instance and verifies the round trip:

```python
import tiga as tg

with tg.runtime.HierarchyRuntime(budgets={"ram": 16, "nvme": 8}) as rt:
    hot = rt.allocate("state", 7, tier="ram", capacity_bytes=8)
    hot.buffer.write(b"tiga1234")
    cold = rt.allocate("state", 7, tier="nvme", capacity_bytes=8)
    rt.transfer(hot, cold).wait()
    hot.close()
    back = rt.allocate("state", 7, tier="ram", capacity_bytes=8)
    rt.transfer(cold, back).wait()
    assert back.buffer.read() == b"tiga1234"
```

The runtime context owns these physical instances and closes them on exit. In contrast, an execution policy context does not invalidate returned public Tensors. RAM↔NVMe file copies use bounded chunks, and device copies use native stream/event ordering. No compiler automatically inserts all necessary spill/reload decisions for an arbitrary program.

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
mesh = tg.DeviceMesh("cuda", (2, 4), names=("rack", "gpu"))
graph = graph.halo(mesh, partition=tg.ByDestination(mesh_axis="gpu"), depth="auto")
```

### Ownership and ghosts, exactly

With `N` destination entities and `W` ranks, the default balanced partition
gives rank `r` the contiguous row range

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

![Halo exchange followed by computation of all owned rows](assets/halo-exchange.svg)

### Forward: communication then compute { #forward-interiorboundary-split-and-overlap }

Distributed forward execution follows this sequence:

1. Pack and exchange halo values from owner to ghost.
2. Wait for the received source values to be ready.
3. Compute all owned destination rows in one local call.

All owned rows write to one local output. The graph and edge function do not
require a scheduling option. CPU, MPI, TCP and NCCL follow
the same dependency order; transport selection only changes data movement.

Every call records `DistributedRuntime.last_execution_trace` (schema
`tiga.distributed-execution-trace.v1`). Its schedule is
`host-staged-serialized` or `device-direct-serialized`, with
`interior_rows=0` and `measured_overlap_ms=0`.

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

### Two-host CUDA sanity check { #two-host-cuda-sanity }

[multi_host_gpu_gate.py](https://github.com/walkerchi/TIGA-lang/blob/main/benchmarks/distributed/multi_host_gpu_gate.py)
checks forward, VJP and repeated calls with changed inputs on both ranks. Run from each host's repository root:

```bash
# Host A: substitute A's reachable private address.
PYTHONPATH=python python benchmarks/distributed/multi_host_gpu_gate.py \
  --rank 0 --host 192.168.1.10 --port 29570 --device cuda:0 --output output/rank0.json

# Host B: connect to A; both hosts can use local device cuda:0.
PYTHONPATH=python python benchmarks/distributed/multi_host_gpu_gate.py \
  --rank 1 --host 192.168.1.10 --port 29570 --device cuda:0 --output output/rank1.json
```

Both files must report `correct=true`. Matching source, built compiler tools,
and compatible Triton and CUDA are required; GPU architectures can differ.
Global rank identifies the participating process; the local CUDA ordinal
selects its GPU. See [distributed performance measurements](experiments.md)
for latency across partition sizes.

Halo derivation and communication-then-compute execution are automatic, not host discovery, remote process launch or GPU-speed-aware load balancing. TCP stages through host memory, not NCCL/GPU-direct. Object messages use pickle: connect trusted private peers only. `timeout` bounds connection and data waits; allow sufficient cold-JIT time. A full mesh with more than two ranks requires reachable peer listening ports.

### NCCL on two hosts { #two-host-nccl }

Use `--transport nccl` to check graph forward, VJP and repeated calls on the
NCCL path. Startup distributes the communicator ID over TCP; NCCL then
exchanges device buffers. For a standalone bidirectional 1 MiB exchange, run
the [NCCL probe](https://github.com/walkerchi/TIGA-lang/blob/main/benchmarks/distributed/multi_host_nccl_probe.py).

The following command selects Socket communication. Replace `INTERFACE` with
a network interface reachable by both hosts and `HOST` with rank 0's address:

```bash
# Run on each host with distinct RANK=0/1 and a shared reachable HOST.
NCCL_SOCKET_IFNAME=INTERFACE NCCL_SOCKET_FAMILY=AF_INET NCCL_IB_DISABLE=1 \
  PYTHONPATH=python timeout 90s python benchmarks/distributed/multi_host_gpu_gate.py \
  --rank RANK --host HOST --port 29659 --transport nccl --output rank-RANK.json
```

Both records must report `correct=true` and agree on source and NCCL library
hashes. Use a launcher timeout to bound initialization waits and expose
transport ports only to trusted peers.

### Runtimes and transports

Communication is a runtime stage, separate from local kernel compilation. CPU
halo VJP materializes its cotangent, exchanges contributions, then exposes the
result as an input to the next LLVM-compiled operation, including in strict
native mode. The active runtime must outlive gradient realization.

### Bind a runtime { #bind-runtime }

Execution requires an active runtime; calling a distributed graph without one
fails closed (`NotImplementedError`):

```python
with tg.distributed.DistributedRuntime(transport):
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

Persistent topology keeps the same API: `tg.save(graph, "mesh.gfg")` writes a
versioned paged store (manifest + `row_ptr.bin` + `col_idx.bin`), and
`tg.load("mesh.gfg").halo(mesh, depth="auto")` returns to the same partitioned
snapshot with each rank reading only its own pages. `tg.load` reads the
manifest only; `resolve_csr()` remains an explicit debug materialization.

Measured transfer, halo-exchange, overlap and NCCL evidence for these paths is
collected on the [benchmark results](benchmark-results.md) page.
