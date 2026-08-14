"""Torch-independent cuBLAS binding used by compiler library dispatch."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from threading import Lock
from typing import Callable

from . import Buffer, Device, DeviceType, Stream


_CUBLAS_STATUS_SUCCESS = 0
_CUBLAS_OP_N = 0
_CUDA_R_16F = 2
_CUBLAS_COMPUTE_32F = 68
_CUBLAS_GEMM_DEFAULT = -1
_CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1


class _MatmulAlgo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", _MatmulAlgo),
        ("workspace_size", ctypes.c_size_t),
        ("state", ctypes.c_int),
        ("waves_count", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


def _address(value) -> int:
    if isinstance(value, Buffer):
        return value.address
    pointer = getattr(value, "data_ptr", None)
    if callable(pointer):
        return int(pointer())
    raise TypeError("cuBLAS arguments must be GraphForge buffers or data_ptr objects")


def _load_cublas() -> ctypes.CDLL:
    failures = []
    for name in ("libcublas.so", "libcublas.so.13", "libcublas.so.12"):
        try:
            library = ctypes.CDLL(name)
            break
        except OSError as error:
            failures.append(str(error))
    else:
        raise RuntimeError("cuBLAS library is unavailable: " + "; ".join(failures))
    library.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.cublasCreate_v2.restype = ctypes.c_int
    library.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    library.cublasDestroy_v2.restype = ctypes.c_int
    library.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.cublasSetStream_v2.restype = ctypes.c_int
    library.cublasGemmEx.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]
    library.cublasGemmEx.restype = ctypes.c_int
    return library


def _load_cublas_lt() -> ctypes.CDLL:
    failures = []
    for name in ("libcublasLt.so", "libcublasLt.so.13", "libcublasLt.so.12"):
        try:
            library = ctypes.CDLL(name)
            break
        except OSError as error:
            failures.append(str(error))
    else:
        raise RuntimeError("cuBLASLt library is unavailable: " + "; ".join(failures))
    library.cublasLtCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    library.cublasLtCreate.restype = ctypes.c_int
    library.cublasLtDestroy.argtypes = [ctypes.c_void_p]
    library.cublasLtDestroy.restype = ctypes.c_int
    library.cublasLtMatmulDescCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int]
    library.cublasLtMatmulDescCreate.restype = ctypes.c_int
    library.cublasLtMatmulDescDestroy.argtypes = [ctypes.c_void_p]
    library.cublasLtMatmulDescDestroy.restype = ctypes.c_int
    library.cublasLtMatrixLayoutCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int64]
    library.cublasLtMatrixLayoutCreate.restype = ctypes.c_int
    library.cublasLtMatrixLayoutDestroy.argtypes = [ctypes.c_void_p]
    library.cublasLtMatrixLayoutDestroy.restype = ctypes.c_int
    library.cublasLtMatmulPreferenceCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p)]
    library.cublasLtMatmulPreferenceCreate.restype = ctypes.c_int
    library.cublasLtMatmulPreferenceDestroy.argtypes = [ctypes.c_void_p]
    library.cublasLtMatmulPreferenceDestroy.restype = ctypes.c_int
    library.cublasLtMatmulPreferenceSetAttribute.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    library.cublasLtMatmulPreferenceSetAttribute.restype = ctypes.c_int
    library.cublasLtMatmulAlgoGetHeuristic.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(_HeuristicResult), ctypes.POINTER(ctypes.c_int),
    ]
    library.cublasLtMatmulAlgoGetHeuristic.restype = ctypes.c_int
    library.cublasLtMatmul.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
    ]
    library.cublasLtMatmul.restype = ctypes.c_int
    return library


@dataclass(frozen=True)
class CUDABlasProvider:
    name: str = "cuda.cublas"

    def display_name(self) -> str:
        return "CUDA cuBLAS runtime library"

    def cache_key(self) -> tuple[str, ...]:
        return (self.name, "gemm-ex-f16-f32-accumulate")


class CUDABlasMatmul:
    """One row-major FP16 GEMM executable bound to a CUDA stream."""

    def __init__(
        self,
        *,
        device: Device | str,
        m: int,
        n: int,
        k: int,
        stream_provider: Callable[[], Stream] | None = None,
    ) -> None:
        self.device = Device.parse(device)
        if self.device.type != DeviceType.CUDA:
            raise ValueError("cuBLAS matmul requires a CUDA device")
        self.m, self.n, self.k = m, n, k
        self._library = _load_cublas_lt()
        self._handle = ctypes.c_void_p()
        self._check(
            self._library.cublasLtCreate(ctypes.byref(self._handle)),
            "cublasLtCreate",
        )
        self._operation = ctypes.c_void_p()
        self._a_layout = ctypes.c_void_p()
        self._b_layout = ctypes.c_void_p()
        self._c_layout = ctypes.c_void_p()
        self._check(self._library.cublasLtMatmulDescCreate(
            ctypes.byref(self._operation), _CUBLAS_COMPUTE_32F, 0),
            "cublasLtMatmulDescCreate")
        # Column-major view of row-major C^T=B^T@A^T.
        self._check(self._library.cublasLtMatrixLayoutCreate(
            ctypes.byref(self._a_layout), _CUDA_R_16F,
            self.n, self.k, self.n), "cublasLtMatrixLayoutCreate(A)")
        self._check(self._library.cublasLtMatrixLayoutCreate(
            ctypes.byref(self._b_layout), _CUDA_R_16F,
            self.k, self.m, self.k), "cublasLtMatrixLayoutCreate(B)")
        self._check(self._library.cublasLtMatrixLayoutCreate(
            ctypes.byref(self._c_layout), _CUDA_R_16F,
            self.n, self.m, self.n), "cublasLtMatrixLayoutCreate(C)")
        self._preference = ctypes.c_void_p()
        self._check(self._library.cublasLtMatmulPreferenceCreate(
            ctypes.byref(self._preference)), "cublasLtMatmulPreferenceCreate")
        workspace_limit = ctypes.c_size_t(32 << 20)
        self._check(self._library.cublasLtMatmulPreferenceSetAttribute(
            self._preference, _CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            ctypes.byref(workspace_limit), ctypes.sizeof(workspace_limit)),
            "cublasLtMatmulPreferenceSetAttribute")
        heuristic = _HeuristicResult()
        returned = ctypes.c_int()
        self._check(self._library.cublasLtMatmulAlgoGetHeuristic(
            self._handle, self._operation,
            self._a_layout, self._b_layout, self._c_layout, self._c_layout,
            self._preference, 1, ctypes.byref(heuristic),
            ctypes.byref(returned)), "cublasLtMatmulAlgoGetHeuristic")
        if returned.value != 1 or heuristic.state != _CUBLAS_STATUS_SUCCESS:
            raise RuntimeError("cuBLASLt found no legal matmul algorithm")
        self._algorithm = heuristic.algo
        self._workspace = (
            Buffer(int(heuristic.workspace_size), device=self.device)
            if heuristic.workspace_size else None
        )
        self._stream_provider = stream_provider
        self._owned_stream = Stream(self.device) if stream_provider is None else None
        self._lock = Lock()
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status != _CUBLAS_STATUS_SUCCESS:
            raise RuntimeError(f"{operation} failed with cuBLAS status {status}")

    def current_stream(self) -> Stream:
        selected = (
            self._owned_stream
            if self._stream_provider is None
            else self._stream_provider()
        )
        if selected is None:
            raise RuntimeError("cuBLAS executable has no stream")
        return selected

    def launch(
        self, lhs, rhs, output, *, stream: Stream | None = None,
        record_event: bool = True,
    ):
        selected = stream or (
            self.current_stream()
        )
        if selected is None or selected.device != self.device:
            raise ValueError("cuBLAS stream and executable device differ")
        # cuBLAS is column-major. Row-major C=A@B is the same storage as
        # column-major C^T=B^T@A^T, so swap operands and M/N without copies.
        with self._lock:
            stream_address = selected.address
            self._check(
                self._library.cublasLtMatmul(
                    self._handle,
                    self._operation,
                    ctypes.byref(self._alpha),
                    ctypes.c_void_p(_address(rhs)),
                    self._a_layout,
                    ctypes.c_void_p(_address(lhs)),
                    self._b_layout,
                    ctypes.byref(self._beta),
                    ctypes.c_void_p(_address(output)),
                    self._c_layout,
                    ctypes.c_void_p(_address(output)),
                    self._c_layout,
                    ctypes.byref(self._algorithm),
                    None if self._workspace is None else ctypes.c_void_p(
                        self._workspace.address),
                    0 if self._workspace is None else self._workspace.nbytes,
                    ctypes.c_void_p(stream_address),
                ),
                "cublasLtMatmul",
            )
        return selected.record_event() if record_event else None

    def prepare(self, lhs, rhs, output, *, stream: Stream | None = None):
        """Prebind pointers and stream to one minimal hot submission closure."""
        selected = stream or self.current_stream()
        stream_pointer = ctypes.c_void_p(selected.address)
        lhs_pointer = ctypes.c_void_p(_address(lhs))
        rhs_pointer = ctypes.c_void_p(_address(rhs))
        output_pointer = ctypes.c_void_p(_address(output))
        workspace_pointer = (
            None if self._workspace is None
            else ctypes.c_void_p(self._workspace.address)
        )
        workspace_size = 0 if self._workspace is None else self._workspace.nbytes
        function = self._library.cublasLtMatmul
        handle, operation = self._handle, self._operation
        a_layout, b_layout, c_layout = (
            self._a_layout, self._b_layout, self._c_layout)
        algorithm = ctypes.byref(self._algorithm)
        alpha, beta = ctypes.byref(self._alpha), ctypes.byref(self._beta)

        def submit():
            status = function(
                handle, operation, alpha,
                rhs_pointer, a_layout, lhs_pointer, b_layout, beta,
                output_pointer, c_layout, output_pointer, c_layout,
                algorithm, workspace_pointer, workspace_size, stream_pointer,
            )
            if status != _CUBLAS_STATUS_SUCCESS:
                raise RuntimeError(
                    f"cublasLtMatmul failed with cuBLAS status {status}")

        return submit

    def close(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle and handle.value:
            for layout_name in ("_a_layout", "_b_layout", "_c_layout"):
                layout = getattr(self, layout_name, None)
                if layout and layout.value:
                    self._library.cublasLtMatrixLayoutDestroy(layout)
                    setattr(self, layout_name, ctypes.c_void_p())
            if self._operation and self._operation.value:
                self._library.cublasLtMatmulDescDestroy(self._operation)
                self._operation = ctypes.c_void_p()
            if self._preference and self._preference.value:
                self._library.cublasLtMatmulPreferenceDestroy(self._preference)
                self._preference = ctypes.c_void_p()
            if self._workspace is not None:
                self._workspace.close()
                self._workspace = None
            self._library.cublasLtDestroy(handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["CUDABlasMatmul", "CUDABlasProvider"]
