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

Reserved for register/shared/HBM/RAM/NVMe placement, async-copy pipeline,
working-set and out-of-core experiments. A benchmark enters this directory only
after the corresponding storage/event semantics exist in compiler IR.
