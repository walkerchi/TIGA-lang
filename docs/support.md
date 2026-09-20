# Troubleshooting and reporting bugs

Tiga is alpha. A constructor's availability does not establish support for every
dtype, layout, gradient, device and provider. Consult the [support matrix](roadmap.md),
[execution diagnostics](execution.md) and [typed API reference](api.md).

## Installation { #installation }

Official release targets: [GitHub](https://github.com/walkerchi/TIGA-lang) for
code/issues, PyPI for `tiga-lang`, and a future arXiv technical report. The first
PyPI release and arXiv identifier are pending. This documentation host is an
authenticated preview, not yet a public documentation service.

Matching wheels need no separate LLVM SDK. Local validation covers Linux x86-64,
CPython 3.11/3.12 and glibc 2.38 or newer. Other Python versions and platforms
are outside the first release. The optional adapter targets Torch 2.11.x and
Triton 3.6.x; the base wheel still has no Torch dependency.
Do not rename wheels to bypass pip's compatibility checks.

| Symptom | Action |
|---|---|
| `No matching distribution found` | Check publication status, Python/OS/architecture and wheel tag; use a matching local wheel or the source build. Do not install an unrelated `tiga` package. |
| pip starts a CMake build | A compatible wheel was not selected. Use `--only-binary=:all:` to require a wheel, or deliberately prepare the pinned source toolchain. |
| Missing Torch | Install Torch separately for Torch examples, or use the native CPU example in the execution guide. |
| Native tools not found | Run `python -m tiga`; check the interpreter and stale `TIGA_OPT` / `TIGA_TRANSLATE` overrides. Wheels bundle these tools. |
| CUDA fails | Check driver/device and the `cuda` extra. Installing Tiga does not install drivers or choose a matching Torch build. |
| Missing visualization dependency | Install `tiga-lang[visualization]`; MP4 also needs external `ffmpeg`, OpenVDB export needs `pyopenvdb`. |

## Input and execution pitfalls { #input-contracts }

| Situation | Contract / next check |
|---|---|
| Reversed CSR edges | Row i is destination i; `col_idx` contains source IDs. See [the connection diagram](programming-model.md). |
| Empty rows / isolated nodes | Repeated row boundaries represent empty destinations. Pass `num_src` to retain isolated sources; check the reducer's empty-row semantics. |
| Duplicate COO/CSR edges | Each edge is a separate message, not deduplicated. COO import sorts by destination; align edge fields with the resulting order. |
| External or hand-built CSR | Use `validate="full"` for boundary/range validation; `basic` is not exhaustive. |
| Bipartite graph | Bind separate `src` and `dst` with leading dimensions `num_src` and `num_dst`; avoid homogeneous `ndata`. |
| Device or layout mismatch | Keep fields and indices on the graph device; follow each operator's dtype/layout constraints. Storage dtype support is not kernel support. |
| Changing topology | Rebuild the graph; do not mutate captured index buffers. Transpose/cat produces a new edge order: old edge fields need reordering. |
| Missing gradients | Set `requires_grad` before calling, and use the matching autograd system. `tg.from_torch` shares storage but not Torch/native autograd history. |
| Dynamic neighbors / pruning | Gradients concern the selected fixed topology, not discrete neighbor selection. Approximate tile-pruned attention is forward-only. |
| Prepared launches | `prepare()` is path-specific and binds fixed storage; not every backend supports it. Automatic whole-Tensor LRU eviction is incompatible. |
| `MemoryError` with free RAM | A scoped allocation budget may be exceeded. Reports exclude Torch allocations and whole-process RSS. |
| Multi-rank hang / timeout | Start every rank with matching collective order; verify endpoints and finite timeout, including cold JIT. DeviceMesh launches no workers. |
| Multiprocessing after GPU/thread initialization | Use a `spawn` context and an `if __name__ == "__main__":` entry guard, as in the distributed example; do not fork an initialized CUDA runtime. Launch all ranks before waiting for results. |

Compare a small case with `program.reference(...)` and explicit tolerances in
`torch.testing.assert_close`. Matching forward values do not validate gradients,
prove compilation or guarantee bitwise determinism. Higher-order gradients and
concurrent mutation are not general supported contracts.

## Reporting a bug { #bug-report }

Ordinary bugs and documentation requests belong in [GitHub Issues](https://github.com/walkerchi/TIGA-lang/issues/new/choose)
once the official repository is public. Sole maintainer: **walkerchi**,
Independent Developer. Email support:
[walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com), subject prefix
`[Tiga bug]`. When issues are unavailable, send the report by email. Include:

1. Expected versus actual behavior and the complete exception.
2. Standalone, seeded reproduction with synthetic data and no private dependencies.
3. Environment, installation command, source commit or wheel filename; GPU/driver if relevant.
4. Graph kind, sizes, dtype/device, forward/backward and `program.explain()`; attach available IR with private paths/data removed.
5. For gradients: oracle, loss/cotangent, tolerances and errors. For performance: matched baseline and raw timings separating build/compile/transfer/warm execution, following [the protocol](performance.md).

```bash
python -m tiga
python -m pip show tiga-lang torch triton
python -m pip check
```

An absent optional package in `pip show` is not an installation failure.
Fixes should add regression tests and update both language pages.

## Security { #security }

Do not post vulnerabilities or private crash dumps in public issues. Follow
[SECURITY.md](https://github.com/walkerchi/TIGA-lang/blob/main/SECURITY.md).
Send private reports to [walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com)
with subject prefix `[Tiga security]`. No fixed response-time SLA is promised.
Tiga is not a sandbox for untrusted code or files.

## Publication and paper { #publication }

The local `tiga-lang-paper` project contains the tech-report draft; no arXiv ID
or paper DOI exists yet. Use [software citation metadata](https://github.com/walkerchi/TIGA-lang/blob/main/CITATION.cff)
until a paper record is available.

Public release requires accessible public docs, monitored support/security,
matching source/wheel versions, clean-environment tests, benchmark source provenance
and tagged artifacts. Local tests and historical charts do not replace this checklist.
