# Memory and storage

Start with ordinary [Tensor](https://en.wikipedia.org/wiki/Tensor_(machine_learning)) operations. Memory policy belongs around the program, not inside its message function.

## A complete example

The example needs the native CPU runtime; Torch is not required. It verifies the gradient after a spill and a persistent value round trip.

```python
--8<-- "examples/hierarchical_memory.py:core"
```

Run [`python examples/hierarchical_memory.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/hierarchical_memory.py). Expected gradient: `[0, 2, 4, 6, 8, 10, 12, 14]`.

## Four separate operations

| Intent | API | Contract |
|---|---|---|
| Set allocation policy | `tg.execution(device=..., memory=..., spill_dir=...)` | Defaults and accounting for allocations created in this scope |
| Change residency | `x.spill()` / `x.realize()` | Same logical Tensor, device, value and autograd history; temporarily stored on disk |
| Copy to a device | `y = x.to("cpu")` / `x.cpu()` | Same device returns the realized input; different device returns a differentiable copy; input device unchanged |
| Persist a value | `tg.save(x, path)` / `tg.load(path)` | Explicit snapshot path; no eviction, no saved autograd graph |

A spilled CUDA Tensor still has a CUDA logical device. `realize()` restores it there; `cpu()` returns a CPU copy. `x.residency` reports `device`, `resident`, `backing`, `bytes` and `version`. Residency is physical placement, not a change to the mathematical value.

## Budget and lifetime

`memory={"host": "1GiB", "device": "4GiB", "nvme": "8GiB"}` accepts non-negative integer bytes or integer IEC sizes (`KiB/MiB/GiB/TiB`). `host` aliases `ram`, `hbm` aliases `device`, and `ssd` aliases `nvme`. Unspecified tiers are unlimited. Pinned allocations have their own `host-pinned` budget.

The hard check covers live native buffers and temporary spill payloads **created in the context**, including CPU page-prefetch workers. A shared buffer is charged once and remains charged while a view owns it. A spill therefore does not promise to free bytes still held by a view, a prepared launch or another owner. A budget failure raises `MemoryError` before the new managed allocation.

`run.memory_report()` returns per-tier live, peak and budget bytes, plus the accounting scope and exclusions. It is **not a process [RSS](https://en.wikipedia.org/wiki/Resident_set_size) limit**: external Torch storage, earlier allocations, Python/compiler memory, OS page cache, bounded file-I/O staging and persistent snapshot files are excluded. File transfers use chunks up to 1 MiB; non-contiguous saves gather logical elements and can be slower.

Exiting the context resets defaults; returned tensors stay valid and retain their original accounting owner. Nested execution contexts are rejected. Anonymous spills are removed on restore or Tensor collection. Explicit snapshots remain until removed by the application. `save` refuses an existing path unless `overwrite=True`; an already attached reader retains its original snapshot when Tiga atomically replaces that path.

## Large graphs: current boundary

`page_rows` and `prefetch_depth` on `tg.execution` set defaults for `paged_csr` calls; explicit call arguments win. Page height is a row count, **not a byte-budget-derived schedule**. Smaller pages and shallower prefetch can reduce working-set pressure.

CPU supports paged forward and VJP. CUDA supports FP32 paged forward
with disk-backed fields: `tg.load(path, device="cuda")` keeps topology on disk,
and `graph.fields("src")` creates allocation-free field handles on that logical
device. Handles retain their readers even after the original Graph is released.
Each page is gathered in host memory and copied to CUDA. A completed output page
passes through host staging and is written immediately into its final row range;
the executor does not collect a full second copy of the output on the host.
This is **not GPUDirect Storage** or asynchronous
GPU/IO overlap. The full output stays resident; recurrent execution also needs
the previous output. CUDA paged backward still raises `NotImplementedError`.
Run the complete, oracle-checked
[`python examples/paged_cuda.py`](https://github.com/walkerchi/TIGA-lang/blob/main/examples/paged_cuda.py)
example; it needs NumPy and the native CUDA provider.

Compiled code caches retain code independently of invocation inputs. Before a
budget rejection or LRU spill, unreachable Python cycles are collected; collection
can add latency. Budget accounting still excludes allocator pools and compiler
resources and therefore is not a physical VRAM cap.

The RAM limit is also the default topology-offload threshold, unless the legacy `tg.runtime.auto_offload` context overrides it. CSR builders still construct their input before deciding to offload: a graph larger than available managed RAM can fail during construction. Open an existing `.gfg` for bounded topology reads; the current constructor is not an external-memory graph builder. See the [paged graph example](examples/distributed-memory.md#a-disk-resident-giant-graph).

## Billion-edge capacity evidence { #billion-edge-capacity }

The September 19, 2026 experiment executes **1 billion explicit directed edges**
on an RTX 5070 Ti with 16 GiB memory: 62.5 million nodes, 16 neighbors per row,
and 32 FP32 features. Every edge participates in neighbor averaging, and every
output value is checked. This is a regular periodic graph, not a power-law graph.

Complete CSR, input and output residency would require at least **22.82 GiB**
(18.86 GiB even with 32-bit indices). Paging keeps the 7.45 GiB output and bounded
page buffers on the GPU; source fields and topology are read from an NVMe store
through host memory. The fixture writer emits bounded chunks directly into the
existing storage format; it is a benchmark utility, not a new Graph constructor.
Workers use a 12 GiB native-buffer safety budget, with existing services using
about 3 GiB. The capacity comparison is against physical GPU memory, not this budget.

[![Billion-edge capacity and complete forward time](assets/results/billion-capacity.png)](assets/results/billion-capacity.png)

The left panel compares a **calculated full-residency lower bound** with measured
native allocation peaks; it is not a measured resident OOM. The right panel shows
complete forward time, including paging, transfers, JIT and output assembly.
Points are medians of three fresh processes; bars show their range. Construction
and full output checking are outside forward timing. OS and compiler caches are
not cleared, and existing GPU services remain active.

At 1B edges, median forward time is **414.25 s** (411.36–415.61 s).
The maximum native GPU allocation peak across trials is **10.27 GiB**, including
the output; forward process peak RSS is 8.83 GiB. These are different memory accounts.

[Raw trials](assets/results/billion/results.json) ·
[Environment and measurement scope](assets/results/billion/ENVIRONMENT.txt) ·
[Reproduction commands](assets/results/billion/REPRODUCE.txt) ·
[Benchmark source](assets/results/billion/billion_edges.py) ·
[Source snapshot](assets/results/billion/source.tar.gz)

This establishes single-GPU forward capacity, **not a speedup or training claim**.
The graph fits host RAM. A billion-edge recurrent step needs space for both its
previous field and next output and is not established by these one-forward trials.

## Automatic eviction of idle Tensors { #automatic-eviction }

Opt-in [LRU](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)) spills idle native Tensor values synchronously before allocating over budget. Reads restore them to their original logical device.

```python
import tempfile
import tiga as tg

with tempfile.TemporaryDirectory() as directory:
    with tg.execution(device="cuda:0", memory={"device": 128, "nvme": 1024},
                      spill_dir=directory, eviction="lru") as run:
        x = tg.tensor([2., 3.], requires_grad=True)
        idle = [tg.tensor([10., 20.]) for _ in range(20)]
        assert not x.residency["resident"]
        assert (x * x).tolist() == [4., 9.]
        assert tg.autograd.grad((x * x).sum(), x).tolist() == [4., 6.]
        print(run.memory_report())
        del idle, x  # Release attachments before removing the directory.
```

The tiny budget demonstrates eviction, not performance. For CPU, use `device="cpu"` and `memory={"ram": 128, "nvme": 1024}`.

- Default `eviction="error"` does not evict. `"lru"` requires an explicit `spill_dir` on a disk with sufficient capacity. The `nvme` tier does not detect the storage medium or imply GPUDirect Storage.
- Active inputs are protected until CUDA completion. Transfers are synchronous; I/O/kernel overlap is not promised.
- Eviction handles **whole Tensors**, not tiled kernels. An active input/output/temporary working set that exceeds the budget still raises `MemoryError`. Disk-budget and I/O failures propagate.
- Only scope-owned native allocations are eligible, not external Torch buffers, earlier allocations or fixed-address prepared launches. `prepare()` explicitly fails inside LRU scopes. Views may retain an allocation.
- Reports include `eviction_policy`, `evictions` and `restores`. Budgets are per execution scope, not a shared multi-process/GPU pool; each rank configures its own.

General GPU↔NVMe out-of-core kernel scheduling, byte-budget-derived page sizes and multi-tier asynchronous prefetch remain unsupported.

## Compatibility and compiler internals

`x.disk()` remains an alias-like legacy entry point for temporary spill; named `disk(name=...)` / `tg.from_disk(name)` remain supported, reject name collisions and path separators, and use the legacy shared directory. Prefer explicit `save/load` paths for new persistence code.

Compiler `HierarchyRuntime` owns named/versioned physical instances, not public Tensor identities. It shares allocation accounting and chunked file-transfer primitives with Tensor storage; graph paging is still a distinct execution path. See [compiler memory and distributed execution](memory-and-distributed.md) and the [API contracts](api.md#spill-and-disk).
