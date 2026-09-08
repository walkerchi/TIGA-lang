"""Subprocess smoke tests for the distributed benchmark entry points."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from _distributed_gates import (
    cuda_device_available, mpi_launcher, nccl_skip_reason,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _benchmark_environment() -> dict[str, str]:
    environment = os.environ.copy()
    paths = [str(REPO_ROOT / "python")]
    bindings = REPO_ROOT / "build" / "mlir-22.1.8" / "python_bindings"
    if bindings.is_dir():
        paths.append(str(bindings))
    if environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    return environment


def _run_benchmark(argv, *, timeout):
    return subprocess.run(
        argv, cwd=REPO_ROOT, env=_benchmark_environment(),
        text=True, capture_output=True, timeout=timeout,
    )


def _completed_message(completed) -> str:
    return f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"


class DistributedBenchmarkSmokeTest(unittest.TestCase):
    def test_halo_exchange_benchmark_small(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            completed = _run_benchmark(
                [sys.executable, "-m", "benchmarks.distributed.halo_exchange",
                 "--entities", "4096", "--features", "8", "--repeats", "3",
                 "--output", str(output)],
                timeout=180)
            self.assertEqual(
                completed.returncode, 0, msg=_completed_message(completed))
            result = json.loads(output.read_text())
            self.assertEqual(result["gate"], "PASS")
            self.assertTrue(result["correct"])

    def test_mpi_halo_exchange_benchmark_small(self):
        mpiexec = mpi_launcher()
        if mpiexec is None:
            self.skipTest("mpiexec/mpi4py with an MPI runtime is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            completed = _run_benchmark(
                [mpiexec, "-n", "2", sys.executable, "-m",
                 "benchmarks.distributed.mpi_halo_exchange",
                 "--entities", "4096", "--features", "8", "--repeats", "3",
                 "--output", str(output)],
                timeout=180)
            self.assertEqual(
                completed.returncode, 0, msg=_completed_message(completed))
            result = json.loads(output.read_text())
            self.assertEqual(result["gate"], "PASS")
            self.assertTrue(result["correct"])

    def test_nccl_loopback_benchmark_small(self):
        reason = nccl_skip_reason()
        if reason is not None:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            completed = _run_benchmark(
                [sys.executable, "-m",
                 "benchmarks.distributed.nccl_device_loopback",
                 "--sizes", "4096", "--warmup", "1", "--repeats", "3",
                 "--output", str(output)],
                timeout=300)
            self.assertEqual(
                completed.returncode, 0, msg=_completed_message(completed))
            result = json.loads(output.read_text())
            self.assertEqual(result["gate"], "PASS")
            self.assertTrue(all(
                case["correct"] for case in result["cases"]))

    def test_nccl_two_gpu_gate_refuses_single_gpu(self):
        if cuda_device_available("cuda:1"):
            self.skipTest("two CUDA devices are available")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            completed = _run_benchmark(
                [sys.executable, "-m", "benchmarks.distributed.nccl_two_gpu_gate",
                 "--bytes", "4096", "--repeats", "2",
                 "--output", str(output)],
                timeout=300)
            self.assertNotEqual(
                completed.returncode, 0, msg=_completed_message(completed))
            # The gate must fail before manufacturing any artifact.
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_automatic_overlap_quick(self):
        if os.environ.get("TIGA_BENCHMARK_SMOKE") != "1":
            self.skipTest(
                "heavy overlap smoke requires TIGA_BENCHMARK_SMOKE=1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.json"
            completed = _run_benchmark(
                [sys.executable, "-m", "benchmarks.distributed.automatic_overlap",
                 "--quick", "--output", str(output)],
                timeout=900)
            self.assertEqual(
                completed.returncode, 0, msg=_completed_message(completed))
            result = json.loads(output.read_text())
            self.assertEqual(result["gate"], "PASS")
            self.assertGreater(result["median_measured_overlap_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
