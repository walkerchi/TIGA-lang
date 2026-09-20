# Distributed benchmarks

`halo_exchange.py` is the core transport conformance benchmark: it launches
two independent processes, derives exact owner/ghost maps from one CSR
snapshot, and measures pack/duplex exchange/unpack with byte-exact validation.
The stdlib pipe provider establishes semantics and regression coverage; NCCL,
MPI and RCCL providers must use the same workload and report their own peer.

The sole public execution policy is **communication then compute**. Transports
cannot automatically enable overlap. `paper_scaling.py` measures single-GPU
baselines and this one distributed policy. Older raw archives remain unchanged.

`automatic_overlap.py` is a historical internal regression probe using
an explicit private override, not a public scheduling option. It exercises `Graph.halo()`
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

## Real two-host scaling

The current spatial experiment uses `--transport nccl --topology mesh
--mesh-sides 32,64,96`, with 16 features. Each graph is a nonperiodic hexahedral
mesh split across a spatial plane, with interface-sized halos rather than
volume-sized ghost sets. Two warmups and five checked calls compare
both single-GPU baselines with communication-then-compute (`serialized`)
two-rank execution. The archived `automatic` rows are historical overlap probes. The
public [three-experiment description](../../docs/experiments.md) states the
negative scaling result and links complete artifacts. Formal raw records are
under `output/paper-three-questions/distributed-spatial`; ring/large-halo records
remain under `distributed-final`, and earlier TCP evidence below
is retained under its original source and timing boundary.

`paper_scaling.py` runs the full graph on each GPU separately, then both real
hosts with serialized and automatic halo paths. Launch with `--rank 0` and
`--rank 1`, the same reachable private `--host`, `--port`, `--sizes`, `--features`
and repeat count, and separate `--output` files. The default uses degree 16,
25% boundary rows, 16 FP32 features, 3 warmups and 10 checked samples.

Both artifacts are required. Verify Python and compiler hashes match, then take
the maximum of the two rank durations **per sample** before computing a median.
The September 19 RTX 5070 Ti + RTX 4070 Ti SUPER TCP run was slower than either
single GPU. Recorded socket/interior host intervals overlap and must not be
stacked as mutually exclusive GPU costs. Setup topology is replicated on each
host; this is not a distributed topology-capacity experiment.

`multi_host_nccl_probe.py` bootstraps NCCL over TCP and checks a real 1 MiB
exchange. Run each rank under an external timeout: communicator setup can block
inside vendor code. A successful TCP control connection does not establish that
NCCL's independently selected peer addresses are routable through WSL/NAT.

The September 19 follow-up passed the 1 MiB probe and
`multi_host_gpu_gate.py --transport nccl` on the two real GPUs. It covers
serialized/automatic/changed-input forward and VJP, not NCCL scaling. An isolated
WireGuard interface used MTU 1200 beneath WSL's outer MTU 1280; a larger tunnel
MTU caused bootstrap packet retransmissions even though small pings succeeded.
Set `NCCL_SOCKET_IFNAME` to the reachable peer interface and check both result
files' source/library hashes. NCCL's Socket plugin here is not GPUDirect RDMA.
