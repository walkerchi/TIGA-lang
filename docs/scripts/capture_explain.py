"""Capture real `kernel.explain()` output for every example, on CPU and CUDA.

Each (example, mode) pair runs in a subprocess for isolation. The runner
patches device defaults so the SAME example source executes on the requested
device, then prints the explain() text of every kernel instance that recorded a variant
during the run. Output files land in output/inspection/<stem>.<mode>.txt and
are meant to be embedded into the docs as inspection snapshots, not proof of compilation or measured performance.

Usage:
    python docs/scripts/capture_explain.py [example.py ...]
    # no arguments: sweep the default example list
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "inspection"
PY = sys.executable

# mode: "both" (default), "asis" (single run, no device forcing)
SPECIAL = {
    "distributed_halo.py": "asis",       # spawns its own two processes
    "hierarchical_memory.py": "asis",    # RAM/NVMe semantics, CPU-only
    "paged_giant_graph.py": "asis",      # disk-paged execution, CPU-only
}


def runner_source() -> str:
    return r'''
import contextlib
import io
import runpy
import sys

path, mode, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

# The example may import sibling modules (e.g. fem_poisson imports solvers)
# and may parse argv (gpu_heatmap) — isolate both.
sys.path.insert(0, str(__import__("pathlib").Path(path).parent.absolute()))
sys.argv = [path]

import torch

import tiga as tg

seen = []
_orig_record = tg.Kernel._record_variant


def _record(self, variant):
    if not any(existing is self for existing in seen):
        seen.append(self)
    return _orig_record(self, variant)


tg.Kernel._record_variant = _record


def _remap_device(kwargs, target):
    device = kwargs.get("device")
    if device is not None and "cuda" in str(device) and target == "cpu":
        kwargs["device"] = "cpu"
    return kwargs


if mode == "cpu":
    torch.set_default_device("cpu")
    for name in ("tensor", "randn", "randn_like", "zeros", "zeros_like",
                 "ones", "ones_like", "arange", "linspace", "tensor"):
        original = getattr(torch, name, None)
        if original is None:
            continue

        def make(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **_remap_device(kwargs, "cpu"))
            return wrapper

        setattr(torch, name, make(original))
    for name in ("dense", "triangular"):
        original = getattr(tg.Graph, name)

        def make(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **_remap_device(kwargs, "cpu"))
            return classmethod(wrapper)

        setattr(tg.Graph, name, make(original))
elif mode == "cuda":
    torch.set_default_device("cuda")
    _orig_gf_tensor = tg.tensor

    def _gf_tensor_cuda(*args, **kwargs):
        kwargs.setdefault("device", "cuda")
        return _orig_gf_tensor(*args, **kwargs)

    tg.tensor = _gf_tensor_cuda

with contextlib.redirect_stdout(io.StringIO()):
    namespace = runpy.run_path(path, run_name="__main__")

with open(out_path, "w", encoding="utf-8") as handle:
    wrote = False
    for kernel in seen:
        handle.write(f"### {type(kernel).__name__}\n")
        try:
            handle.write(kernel.explain() + "\n\n")
        except Exception as error:  # noqa: BLE001
            handle.write(f"(explain unavailable: {error})\n\n")
        wrote = True
    if not wrote:
        # Program-captured kernel calls record no Kernel variant either; the
        # meaningful artifact is the GraphProgram's explain() plus the fused
        # launch count from the kernel IR.
        from tiga.program import ProgramValue

        reported = set()
        for name, value in sorted(namespace.items()):
            if name.startswith("_") or not isinstance(value, ProgramValue):
                continue
            program = value.program
            if id(program) in reported:
                continue
            reported.add(id(program))
            handle.write(f"### {name}.program: GraphProgram\n")
            handle.write(program.explain() + "\n")
            try:
                launches = program.ir("kernel").count('"gf_kernel.launch"')
                handle.write(f"kernel IR: {launches} x gf_kernel.launch\n")
            except Exception:  # noqa: BLE001
                pass
            handle.write("\n")
            wrote = True
    if not wrote:
        # Tensor/program-level compilations record no Kernel variant; surface
        # any namespace object that can explain itself, plus Tensor execution
        # records (backend, provider artifacts).
        for name, value in sorted(namespace.items()):
            if name.startswith("_"):
                continue
            explain = getattr(type(value), "explain", None)
            if explain is not None and not isinstance(value, tg.Kernel):
                try:
                    handle.write(f"### {name}: {type(value).__name__}\n")
                    handle.write(value.explain() + "\n\n")
                    wrote = True
                except Exception:  # noqa: BLE001
                    continue
        for name, value in sorted(namespace.items()):
            execution = getattr(value, "execution", None)
            if isinstance(execution, dict) and execution:
                handle.write(f"### {name}: execution record\n")
                for key, item in execution.items():
                    handle.write(f"{key}: {item}\n")
                handle.write("\n")
                wrote = True
    if not wrote:
        # Pure Tensor-expression examples compile no kernel object; the
        # established inspection handle is the semantic Tensor IR.
        for name in ("output", "y", "loss", "result", "out"):
            value = namespace.get(name)
            mlir = getattr(value, "mlir", None)
            if callable(mlir):
                try:
                    lines = mlir().splitlines()
                except Exception:  # noqa: BLE001
                    continue
                handle.write(f"### {name}: Tensor IR (first lines)\n")
                handle.write("\n".join(lines[:14]) + "\n")
                if len(lines) > 14:
                    handle.write("…\n")
                wrote = True
                break
    if not wrote:
        handle.write("(no kernel variant recorded)\n")
'''


def capture(example: Path, mode: str, runner: Path) -> tuple[bool, str]:
    OUT.mkdir(parents=True, exist_ok=True)
    out_file = OUT / f"{example.stem}.{mode}.txt"
    cmd = [PY, str(runner), str(example), mode, str(out_file)]
    try:
        result = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=240,
            env={"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        out_file.write_text("(run timed out)\n", encoding="utf-8")
        return False, "timeout"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        out_file.write_text(
            "(this configuration does not run the example)\n"
            + "\n".join(tail) + "\n",
            encoding="utf-8",
        )
        return False, tail[-1] if tail else "error"
    if not out_file.read_text(encoding="utf-8").strip():
        out_file.write_text("(no kernel variant recorded)\n", encoding="utf-8")
        return False, "no kernels"
    return True, "ok"


def main() -> None:
    runner = OUT / "_runner.py"
    OUT.mkdir(parents=True, exist_ok=True)
    runner.write_text(runner_source(), encoding="utf-8")

    if len(sys.argv) > 1:
        examples = [Path(arg) for arg in sys.argv[1:]]
    else:
        examples = sorted(ROOT.glob("examples/*.py")) + sorted(
            ROOT.glob("examples/compiler_probes/*.py"))

    for example in examples:
        style = SPECIAL.get(example.name, "both")
        modes = ("asis",) if style == "asis" else ("cpu", "cuda")
        for mode in modes:
            ok, note = capture(example, mode, runner)
            mark = "OK " if ok else "SKIP"
            print(f"{mark} {example.name}:{mode} ({note})")


if __name__ == "__main__":
    main()
