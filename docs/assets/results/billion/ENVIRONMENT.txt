# Billion-edge experiment environment

Date: 2026-09-19. Local dirty source checkout based on commit
`1c35885e73258105c2e93fb5580c3d7aafd648ec`; not a release tag.
The source snapshot accompanies the measurements; per-worker hashes identify
every imported Tiga Python source and the compiler executables.

- Host: AMD Ryzen 7 255, about 44 GiB RAM, local NVMe filesystem.
- GPU: RTX 5070 Ti, 17094934528 bytes reported capacity, driver 595.84.
- Existing voice service: approximately 3 GiB GPU memory, left running.
- Python: `/home/walkerchi/Code/ComfyUI/.venv/bin/python` (3.12).
- Environment: `PYTHONPATH=python:.`, `TIGA_TENSOR_BACKEND=native`,
  `OMP_NUM_THREADS=1`; no `CUDA_VISIBLE_DEVICES` override.
- `TIGA_OPT=/tmp/tiga-docs-review/compiler/bin/gf-opt`;
  `TIGA_TRANSLATE=/tmp/tiga-docs-review/compiler/bin/gf-translate`.
- Compiler build: Release, LLVM/MLIR 22.1.8, system `/usr/bin/c++`.
- Fixture directory: `/tmp/tiga-billion-20260919` on the NVMe filesystem.
- Three rounds in order: 1B, 10M, 100M edges per round. Each worker is a new
  process with a 1800-second external timeout. GPU benchmark processes do not
  overlap in the final sweep. Existing services remain uncontrolled.
- No OS page-cache clearing or vendor/compiler cache clearing. Measurements
  include cache lookup/JIT as encountered; they are not cold-start compile or
  cold-disk latency measurements. Dataset ingest preceded the sweep.
- Oracle reads the final GPU output in 65,536-row chunks and checks every value.
  The forward timer stops before the oracle begins.

Actual native libraries were verified from the running worker's `/proc/PID/maps`:

| Library | SHA256 |
|---|---|
| `build/cp311-cp311-linux_x86_64/lib/Runtime/libgraphforge_runtime.so` | `84d646d4cf0826c3a827c5e341c7b331dc2bf01c78026c986b84e88d0482c75e` |
| `build/cp311-cp311-linux_x86_64/python_bindings/_graphforge_compiler.cpython-311-x86_64-linux-gnu.so` | `e912da833a32996194286e5d82d9feb3dad82972fbb82bea21e6ddb77dd294ca` |

These are the checkout's existing C-ABI runtime/compiler libraries; no native
library was rebuilt during this sweep. The benchmark uses the native tensor
path and does not import Torch for its computation.

Files named `final-e*-r1.json` through `r3.json` comprise the formal sweep.
The small smoke run, `r0` pilot and interrupted first 1B trial are excluded.
`resident-preflight.json` records an arithmetic capacity rejection without
attempting an impossible full allocation. It is not an observed OOM or timing.
Only complete, zero-error trials are accepted by the paper's import script.
