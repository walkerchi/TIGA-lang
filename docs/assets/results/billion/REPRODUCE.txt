# Memory hierarchy benchmarks

`transfer.py` measures the native CUDA Driver HBM↔page-locked-host path and
the runtime-managed RAM↔NVMe spill path.  It also checks byte-exact round trips
and reports capacity-accounted peak live bytes.  It does not use Torch for
allocation, copies, streams, or timing.

`paged_giant_graph.py` measures the disk-resident paged-CSR MessagePassing
path: one invocation per configuration (fresh process, so `getrusage` peak
RSS is uncontaminated), reporting elapsed time and peak RSS for page-size
and prefetch on/off. Build the cache once with `--build-only`, then run one
process per configuration, e.g. `--page-rows 100000 --prefetch 1`.

`capacity_scaling.py` compares full materialization with CPU paged execution
under a fixed native-buffer RAM budget. The default sweep uses four graph sizes,
two page sizes, and three fresh-process trials per configuration. Each trial
checks one complete forward result against an independent NumPy oracle and
records native allocation peaks, process peak RSS, and load-plus-first-execution
time (including JIT). Resident budget failures remain in the output.

Run from the repository root with the native compiler configured:

```bash
TIGA_TENSOR_BACKEND=native OMP_NUM_THREADS=1 python -m \
  benchmarks.memory_hierarchy.capacity_scaling \
  --cache-dir /path/to/new-fixtures --output output/new-capacity-run
```

The fixture path must be new; existing graphs are not overwritten. The native
budget is **not** a process RSS limit. Ingest, process startup and oracle checking
are outside the timed interval, and the OS page cache is not forcibly cleared.
This is single-invocation CPU capacity evidence, not GPU/NVMe streaming,
cold-disk bandwidth, repeated-training throughput or a complete compiled
Task/Storage IR scheduling experiment.

`gpu_paged_scaling.py` measures actual CUDA paged forward over an existing NVMe
store, including host staging, at 3 graph sizes and 2 page sizes. The default is
24 MiB of native device buffers, 10 recurrent steps and 3 fresh-process trials.
Every step consumes the previous output and is checked against NumPy. The raw
data separates initial loading and per-step execution, and retains resident
budget failures. Explicit between-step GC and oracle checks are excluded from
forward timing; live-memory checkpoints are after GC. This is not GPUDirect
Storage, a hard physical VRAM limit, asynchronous IO overlap or paged training.

```bash
TIGA_TENSOR_BACKEND=native OMP_NUM_THREADS=1 python -m \
  benchmarks.memory_hierarchy.gpu_paged_scaling \
  --cache-dir /path/to/new-gpu-fixtures --output output/new-gpu-capacity-run
```

## Billion-edge CUDA capacity

[`billion_edges.py`](billion_edges.py) builds and executes explicit 10M, 100M and
1B-edge CSR graphs without constructing all indices or output checks in host
memory at once. The 1B case has 62.5M nodes, degree 16 and 32 FP32 features. Its
CSR, input and output require at least 22.82 GiB together (18.86 GiB even with
32-bit indices), whereas paged execution retains the 7.45 GiB output plus page
buffers on the GPU. All stored edges participate in the mean-neighbor forward.
This controlled periodic graph has local gathers; it is not a power-law graph.

Use a source build with the native CUDA provider and configured `TIGA_OPT` and
`TIGA_TRANSLATE` binaries. NumPy and Triton are required. Run from the repository
root with `PYTHONPATH=python:.`, `TIGA_TENSOR_BACKEND=native` and
`OMP_NUM_THREADS=1`. Allow 12 GiB of free GPU memory for the largest case;
allow at least 20 GiB of free host RAM and 20 GiB of disk space. The worker checks
currently available VRAM and preserves a 512 MiB reserve. Existing files are
never overwritten by fixture construction.

```bash
for edges in 10000000 100000000 1000000000; do
  python -m benchmarks.memory_hierarchy.billion_edges --build \
    --edges "$edges" --cache-dir /path/to/new-billion-fixtures
done
for trial in 1 2 3; do
  for edges in 1000000000 10000000 100000000; do
    python -m benchmarks.memory_hierarchy.billion_edges \
      --edges "$edges" --cache-dir /path/to/new-billion-fixtures \
      --output "output/billion/final-e${edges}-r${trial}.json"
  done
done
```

Every worker uses a fresh process and checks **every output element**, including
the wraparound rows, against an independent modular-arithmetic oracle. Forward
time includes page reads, JIT, transfers, computation and incremental output
assembly. Ingest, handle loading and oracle time are separate. Results record
source/compiler/payload hashes, native GPU peaks, process RSS, process I/O
counters and one-second whole-device memory samples. OS caches and compiler
caches are not cleared; this is not cold-NVMe bandwidth or isolated kernel time.
The resident mode rejects a lower bound above physical capacity before allocation
with `infeasible_lower_bound`: this is arithmetic, **not a measured OOM**.
CUDA paged backward and billion-edge recurrent training are outside this test.
