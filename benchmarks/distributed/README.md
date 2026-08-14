# Distributed benchmarks

`halo_exchange.py` is the core transport conformance benchmark: it launches
two independent processes, derives exact owner/ghost maps from one CSR
snapshot, and measures pack/duplex exchange/unpack with byte-exact validation.
The stdlib pipe provider establishes semantics and regression coverage; NCCL,
MPI and RCCL providers must use the same workload and report their own peer.

`automatic_overlap.py` exercises the public `Graph.halo()` MessagePassing path
with both owned-only interior rows and ghost-dependent boundary rows. It writes
`results.json`, a human-readable `REPORT.md`, and a two-rank `timeline.png`;
the gate requires correct output, a nonzero measured intersection between
actual interior Tensor execution and halo transport on every sample, and an
end-to-end win over the same runtime forced to serialize. The default 5 ms
receive delay is an explicit controlled inter-node-latency model, not a claim
about the local pipe provider or a measured network:

```bash
python -m benchmarks.distributed.automatic_overlap --quick
```

The registered full case uses two CPU ranks, N=65,536, degree 16, feature width
64 and a 25% boundary. It records 13.7224 ms median overlap and 38.2262 ms
automatic latency versus 40.8632 ms forced-serialized (1.069x). The source
retains the controlled-delay field in JSON and the report so this cannot be
mistaken for an inter-node measurement.

`mpi_halo_exchange.py` runs the same pack/exchange/unpack boundary over the
optional MPI plugin. Reproduce the registered local-host provider artifact with:

```bash
mpiexec -n 2 python -m benchmarks.distributed.mpi_halo_exchange
```

It is a real two-process MPI measurement, but is deliberately not described as
multi-node or GPU-direct evidence.

`nccl_device_loopback.py` is the local GPU-direct provider gate:

```bash
python -m benchmarks.distributed.nccl_device_loopback
```

It creates a real NCCL communicator and validates native CUDA buffers plus
stream completion. Since a rank cannot be its own NCCL point-to-point peer,
world size one deliberately uses the provider's stream-ordered local D2D path.
Its throughput is not NCCL, multi-GPU, or overlap evidence.

The real X0 hardware gate is intentionally separate and refuses to emit output
unless two CUDA devices exist:

```bash
python -m benchmarks.distributed.nccl_two_gpu_gate
```

It measures peer exchange and verifies `Graph.halo()` MessagePassing forward
plus compiler-generated reverse halo VJP on both devices. A profiler timeline
is still required before claiming communication/computation overlap.

Reserved for partition, halo exchange, interior/boundary splitting, collective
and communication/computation-overlap experiments. Results must report both
local kernel time and end-to-end distributed critical path.
