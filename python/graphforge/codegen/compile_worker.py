"""Persistent out-of-process vendor compiler worker.

The worker warms the vendor's content-addressed disk cache in an isolated
process.  The JIT process subsequently re-enters ``triton.compile`` only to
load the cached executable handle required by the Python launch ABI.  A
compiler crash therefore cannot corrupt the GraphForge runtime process.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from typing import Any, Mapping


def _target_payload(target: Any | None) -> dict[str, object] | None:
    if target is None:
        from triton.runtime import driver

        target = driver.active.get_current_target()
    required = ("backend", "arch", "warp_size")
    if not all(hasattr(target, name) for name in required):
        raise TypeError("vendor target must expose backend, arch and warp_size")
    return {name: getattr(target, name) for name in required}


def _serve() -> int:
    from triton.backends.compiler import GPUTarget
    from triton.compiler import compile as triton_compile

    for line in sys.stdin:
        try:
            request = json.loads(line)
            module = request["module"]
            source_hash = request["source_hash"]
            target_data = request.get("target")
            target = GPUTarget(**target_data) if target_data else None
            started = time.perf_counter_ns()
            with TemporaryDirectory(prefix="graphforge-worker-ttir-") as directory:
                path = Path(directory) / f"{source_hash[:16]}.ttir"
                path.write_text(module)
                # Vendor diagnostics must not corrupt the JSON stdout protocol.
                with contextlib.redirect_stdout(sys.stderr):
                    compiled = triton_compile(
                        str(path), target=target,
                        options=dict(request.get("options") or {}),
                    )
            response = {
                "id": request["id"],
                "ok": True,
                "compile_ms": (time.perf_counter_ns() - started) / 1e6,
                "artifacts": sorted(compiled.asm),
            }
        except BaseException as error:  # keep the worker alive after bad IR
            response = {
                "id": request.get("id") if "request" in locals() else None,
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


class PersistentCompileWorker:
    """Serialized client for one crash-recovering vendor compiler process."""

    def __init__(self, *, timeout_seconds: float = 300.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._next_id = 0

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.poll() is None else None

    def _start(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process
        self._process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        return self._process

    def compile(
        self,
        module: str,
        *,
        source_hash: str,
        target: Any | None,
        options: Mapping[str, Any] | None,
    ) -> dict[str, object]:
        with self._lock:
            process = self._start()
            self._next_id += 1
            request_id = self._next_id
            request = {
                "id": request_id,
                "module": module,
                "source_hash": source_hash,
                "target": _target_payload(target),
                "options": dict(options or {}),
            }
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                self._discard()
                raise RuntimeError("vendor compile worker crashed before request") from error
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            ready = selector.select(self.timeout_seconds)
            selector.close()
            if not ready:
                self._discard(kill=True)
                raise TimeoutError(
                    f"vendor compile worker exceeded {self.timeout_seconds:g}s")
            line = process.stdout.readline()
            if not line:
                code = process.poll()
                self._discard()
                raise RuntimeError(
                    f"vendor compile worker exited unexpectedly (code={code})")
            response = json.loads(line)
            if response.get("id") != request_id:
                self._discard(kill=True)
                raise RuntimeError("vendor compile worker protocol desynchronized")
            if not response.get("ok"):
                raise RuntimeError(
                    "vendor compile worker failed: "
                    f"{response.get('error_type')}: {response.get('error')}")
            return response

    def _discard(self, *, kill: bool = False) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            if kill:
                process.kill()
            else:
                process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        # Popen does not close PIPE file objects when a terminated child is
        # merely waited on.  Close both ends explicitly so crash recovery does
        # not leak descriptors (or emit ResourceWarning during interpreter
        # shutdown and the test suite).
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()

    def simulate_crash_for_test(self) -> None:
        """Terminate the worker; the next request must start a fresh process."""
        with self._lock:
            self._discard(kill=True)

    def close(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is None:
                return
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=2)
            if process.stdout is not None and not process.stdout.closed:
                process.stdout.close()


_WORKER = PersistentCompileWorker(
    timeout_seconds=float(os.environ.get("GRAPHFORGE_COMPILE_TIMEOUT_SECONDS", "300"))
)
atexit.register(_WORKER.close)


def compile_in_worker(
    module: str,
    *,
    source_hash: str,
    target: Any | None,
    options: Mapping[str, Any] | None,
) -> dict[str, object]:
    return _WORKER.compile(
        module, source_hash=source_hash, target=target, options=options)


if __name__ == "__main__":
    raise SystemExit(_serve() if "--serve" in sys.argv[1:] else 2)


__all__ = ["PersistentCompileWorker", "compile_in_worker"]
