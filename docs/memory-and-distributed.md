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

## Hierarchical memory

GraphForge separates a logical value from its physical instances:

```text
Logical Field/Relation
  ├─ register/thread tile
  ├─ shared/LDS/UB workgroup tile
  ├─ device HBM instance
  ├─ host/pinned RAM instance
  ├─ NVMe page stream
  └─ remote partition/replica
```

IR needs to preserve:

- address space and visibility;
- residency, capacity and lifetime;
- tile shape, layout and valid/ragged mask;
- async transfer source/destination;
- producer/consumer event and barrier dependencies;
- buffering count and pipeline stage;
- materialize, evict, spill and recompute legality.

The user does not write `cache(..., space="shared")` in ordinary simulation
code. The compiler derives placement from access/reuse/relation structure and
target capabilities. A future expert schedule API may constrain a decision, but
cannot replace semantic analysis.

## Distributed relations

A distributed graph snapshot carries ownership and partition semantics:

```text
partition relation/fields
  → identify owned and ghost entities
  → pack halo data
  → launch interior work
  → exchange/collective
  → unpack/update ghosts
  → launch boundary work
```

Required IR information includes partition map, owner, ghost version, halo
access set, consistency point, communication group and event dependencies.
This permits legal overlap of interior compute with communication without
changing the user program or observing a half-updated snapshot.

The public API is DTensor-like and remains graph-native:

```python
mesh = gf.DeviceMesh("cuda", (2, 4), names=("rack", "gpu"))
graph = graph.halo(
    mesh,
    partition=gf.ByDestination(mesh_axis="gpu", balance="edges"),
    depth="auto",
)

u_next = Diffusion()(graph=graph, src={"u": u}, dst={"u": u})
```

部署只需在进程初始化时绑定 transport；用户 kernel 不调用通信 primitive：

```python
with gf.distributed.DistributedRuntime(transport):
    u_next = Diffusion()(graph=graph, src={"u": local_u}, dst={"u": local_u})
```

Transport implementations are deployment plugins rather than graph classes:

```python
with gf.distributed.DistributedRuntime.from_provider("mpi"):
    u_next = Diffusion()(graph=graph, src={"u": local_u}, dst={"u": local_u})
```

For NCCL, the launcher/control plane broadcasts the 128-byte communicator ID;
kernel code and graph code are unchanged:

```python
from graphforge.distributed import DistributedRuntime, nccl_unique_id

# Rank zero creates nccl_unique_id(); the launcher broadcasts those bytes.
with DistributedRuntime.from_provider(
    "nccl", rank=rank, world_size=world_size,
    communicator_id=broadcast_id, device=f"cuda:{local_rank}",
):
    u_next = Diffusion()(graph=graph, src={"u": local_u}, dst={"u": local_u})
```

Persistent topology keeps the same graph API:

```python
gf.save(graph, "mesh.gfg")
graph = gf.load("mesh.gfg").halo(mesh, depth="auto")
u_next = Diffusion()(graph=graph, src={"u": local_u}, dst={"u": local_u})
```

`gf.load` reads a manifest only. The distributed planner scans bounded row
partitions for exact send/receive maps and then loads the current rank's row and
edge range. Calling `resolve_csr()` is an explicit debug materialization request.

`Graph.halo()` does not eagerly send data and does not create a
`DistributedGraph` subtype. It attaches ownership/ghost requirements to a new
logical snapshot. The compiler derives the exact remote source set from the
relation and field accesses. Below the user program, `gf.task` schedules pack,
exchange, unpack, interior and boundary kernels. Network communication is not
forced inside one GPU kernel; it is an asynchronous event dependency between
local kernels, which is what permits portable overlap.

## Performance evidence required

`benchmarks/memory_hierarchy/transfer.py` reports transfer volume, achieved bandwidth,
pipeline bubbles, peak residency and end-to-end latency across register/shared/
HBM/RAM/NVMe boundaries. `benchmarks/distributed/halo_exchange.py` reports the real
two-process payload, neighbor exchange latency and effective bandwidth；后续 multi-device
artifact 还要加入 local kernel、pack/unpack、network/collective、exposed communication 和 critical-path；
speedup without the communication boundary is not accepted.
`benchmarks/distributed/automatic_overlap.py` measures that public path against
the same runtime forced to serialize, records per-rank communication/interior
timestamps, and writes `timeline.svg` plus a PNG fallback. Its registered 1.069× result uses an
explicit 5 ms receive-delay model and is therefore scheduler evidence under a
controlled link model, not a measured inter-node claim.
`benchmarks/distributed/nccl_device_loopback.py` separately validates communicator
creation plus the device-buffer provider's local-D2D path and explicitly marks
its rank-one throughput as neither NCCL nor performance evidence.
The device-transport MessagePassing fixture additionally validates the
`enqueue halo → realize interior → wait → boundary` submission order and its
automatic reverse VJP. Host timestamps are reported only as ordering evidence;
actual GPU concurrency requires the two-device profiler gate.
