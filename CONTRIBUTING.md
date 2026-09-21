# Contributing to Tiga

The [API reference](docs/api.md), [support matrix](docs/roadmap.md), and regression
tests describe the current contract. Historical plans live in [docs/archive](docs/archive/README.md).
The [development guide](docs/development.md) contains the source installation,
MLIR test configuration, Python checks and GPU release gate.

## Reproducing a bug

A useful report includes:

1. Environment: `python -m tiga` output, OS, and GPU/driver when relevant.
2. Minimal program: the smallest graph and message function, or tensor expression,
   that reproduces the problem.
3. Diagnostics: the exception and `kernel.explain()`; for native tensors,
   `output.execution` and `output.generated_code()` after realization.

Remove secrets and private data from logs and reproducers.

Ordinary bugs belong in the [official GitHub issue tracker](https://github.com/walkerchi/TIGA-lang/issues/new/choose).
Select the bug, performance or feature/documentation template.
Include expected versus actual behavior, numerical tolerances, the source
revision/wheel filename, and raw measurements for performance reports.
The bilingual [support guide](docs/support.md) covers installation and input
pitfalls. Private vulnerabilities follow [SECURITY.md](SECURITY.md), not public
issues. Contact the sole maintainer, **walkerchi** (Independent Developer), at
[walker.chi.000@gmail.com](mailto:walker.chi.000@gmail.com) for email support.

## Branch and publication workflow

Maintainer changes are pushed to `dev`, then submitted as a pull request into
`main`. Do not push changes directly to `main`. Keep `dev` after merging and
fast-forward it to the updated `main` before the next change. External
contributors can submit pull requests from forks.

Public documentation is available on [GitHub Pages](https://walkerchi.github.io/TIGA-lang/).
The `docs-pages` workflow checks both languages on `dev` and pull requests;
only `main` deploys the public site. Documentation builds do not install Tiga,
Torch or LLVM. Read the Docs configuration remains available for a separate
integration. PyPI publication uses its own explicit release approval; merging
a documentation change does not publish a Python package or an arXiv report.

## Checks and conventions

- Add a regression test for a bug fix and measured evidence for a performance claim.
- Keep the public Python API at the `tiga` package root; subsystems remain packages.
- Keep workload-specific implementations in examples and benchmarks, not compiler code.
- State legality conditions for rewrites. Unsupported execution must report a clear
  error or an explicitly identified reference path.
- Label external-library dispatch and correctness evaluation separately from
  compiler-generated execution.
- Commit intentional, registered benchmark evidence with its provenance; avoid
  unrelated generated output and local build artifacts.

## Documentation

- Keep English and Chinese pages consistent. Use neutral wording, not second person.
- Preserve English technical terms; link the first occurrence to Wikipedia
  (Chinese terms to Baidu Baike).
- Put formulas in standalone `$$` blocks, not inline.
- Link example commands to their source file or the corresponding documented example.
- Give each diagram one clear focus. Inspect rendered desktop and narrow layouts
  after visual changes.
- Run the strict documentation build described in the development guide.
