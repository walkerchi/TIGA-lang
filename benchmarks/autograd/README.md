# Autograd benchmarks

These workloads measure compiler-generated reverse programs against matched
framework and handwritten backward implementations. Handwritten kernels are
oracles under `benchmarks/kernels/`; Tiga runtime never imports them.

Current coverage:

- `message_passing_backward.py`: static-CSR source gradient (`dx`) and scalar
  or vector edge-field gradient (`dweight`) with arbitrary upstream
  cotangent. It reports backward-only latency, checkpoint materialization,
  compiler/provider JIT, roofline and a 95% CI gate separately. Registered
  cases include regular and power-law topology plus F=1/16/64 edge gradients.
- `reducer_product_backward.py`: user-defined non-additive product reducer;
  the compiler proves its monoid, emits a zero-safe CSR VJP, and compares the
  row-tiled TTIR against tuned handwritten Triton and `torch.autograd`.
- `dynamic_radius_backward.py`: a user distance-weighted MessagePassing UDF;
  graph membership is fixed to the forward snapshot while the compiler emits
  position and source gradients through
  `gf_tensor.csr_euclidean_distance_sum_vjp`. The registered
  N=32768/D=3/degree~32 case is 1.079x over matched handwritten Triton (95% CI
  [1.073, 1.082]) and 4.36x over `torch.autograd`.

Pending independent buckets:

- fused multi-field UDFs;
- additive tuple reducers such as mean;
- additional captured reducer algebras beyond the registered product bucket;
- online-softmax saved-state/recompute backward.

Run the formal local bucket:

```bash
PYTHONPATH=python python -m benchmarks.autograd.message_passing_backward \
  --nodes 131072 --degree 16 --repeat 100 \
  --locality random --index-dtype i64 --fail-on-gate
```

Run the dynamic fixed-snapshot bucket:

```bash
PYTHONPATH=python python -m benchmarks.autograd.dynamic_radius_backward \
  --nodes 32768 --dimensions 3 --degree 32 --repeat 100 --fail-on-gate
```
