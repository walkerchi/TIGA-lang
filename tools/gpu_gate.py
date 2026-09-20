"""Pre-release GPU gate: run the CUDA forward/backward/cache/diagnostics suite.

Run on a machine with a real CUDA device before tagging a release:

    PYTHONPATH="$PWD/python" python tools/gpu_gate.py

Exits non-zero when no CUDA device is present (a release gate must not skip
GPU coverage) or when any gated test fails.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

GPU_TESTS = [
    "tests/python/test_doc_examples.py",       # execute CUDA documentation examples
    "tests/python/test_brand_and_readme.py",   # execute both GPU READMEs
    "tests/python/test_release_tensor_safety.py",  # detached aliases / inference mode
    "tests/python/test_radius_hash_grid.py",      # generated index ownership / gradients
    "tests/python/test_ttir_provider.py",        # TTIR/CUDA lowering + provider
    "tests/python/test_attention_examples.py",   # dense/causal/tile attention
    "tests/python/test_edge_nn_tile.py",         # nn-on-edge tile forward
    "tests/python/test_edge_nn_vjp.py",          # nn-on-edge backward
    "tests/python/test_edge_nn_ops.py",          # captured operators and every parameter gradient
    "tests/python/test_edge_nn_attention.py",    # score/value roles and attention VJP
    "tests/python/test_graph_constructors.py",   # cat/transpose/device index boundaries
    "tests/python/test_algorithm_probes.py",     # fused CSR/graph algorithm probes
    "tests/python/test_message_passing_autograd.py",  # forward + cache stats
    "tests/python/test_torch_default.py",  # default Torch fields, gradients, cache
    "tests/python/test_automatic_eviction.py",  # managed paging + native CUDA VJP
    "tests/python/test_graph_program.py",  # inline expansion lifetime regression
    "tests/python/test_multi_host_gpu.py",  # two CUDA ranks, host-staged halo/VJP
]


class ReleaseGate:
    """Reject skipped, deselected or expected-failure release checks."""

    def __init__(self):
        self.incomplete: set[str] = set()

    def pytest_collectreport(self, report):
        if report.skipped:
            self.incomplete.add(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped or hasattr(report, "wasxfail"):
            self.incomplete.add(report.nodeid)

    def pytest_deselected(self, items):
        self.incomplete.update(item.nodeid for item in items)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.incomplete and exitstatus == 0:
            session.exitstatus = 1

    def pytest_terminal_summary(self, terminalreporter):
        if self.incomplete:
            terminalreporter.write_sep("=", "GPU release gate incomplete", red=True)
            for nodeid in sorted(self.incomplete):
                terminalreporter.write_line(nodeid, red=True)


def main() -> int:
    try:
        import torch
    except ModuleNotFoundError:
        print("gpu_gate: torch is required for the CUDA gate", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("gpu_gate: no CUDA device — refusing to pass a release gate "
              "without GPU coverage", file=sys.stderr)
        return 2
    if importlib.util.find_spec("triton") is None:
        print("gpu_gate: Triton is required", file=sys.stderr)
        return 2
    from tiga.compiler.toolchain import find_gf_opt, find_gf_translate

    if not find_gf_opt() or not find_gf_translate():
        print("gpu_gate: built gf-opt and gf-translate are required", file=sys.stderr)
        return 2
    import pytest

    print(f"gpu_gate: {torch.cuda.get_device_name(0)}", flush=True)
    tests = [str(ROOT / name) for name in GPU_TESTS]
    return int(pytest.main(["-q", "-ra", *tests], plugins=[ReleaseGate()]))


if __name__ == "__main__":
    sys.exit(main())
