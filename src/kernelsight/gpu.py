"""Run and time kernels on a real GPU through the CUDA driver API (ctypes).

This module is the optional, GPU-dependent part of KernelSight. It loads the
cubin produced by ``ptxas`` (so the code that runs is exactly the code whose
resource usage was reported), launches a kernel, and times it with CUDA
events.

The CUDA *driver* API lives in ``nvcuda.dll`` (Windows) / ``libcuda.so.1``
(Linux) and is installed with the NVIDIA display driver, so no extra Python
package is needed. Only the handful of calls used below are bound.

Timing method: ``warmup`` untimed launches, then ``repeats`` launches each
bracketed by two events; the report gives the median and interquartile range.
On a laptop GPU, clocks vary with temperature and power, so treat small
differences (a few percent) as noise.
"""

from __future__ import annotations

import ctypes
import statistics
import sys
from dataclasses import dataclass, field

CUresult = ctypes.c_int
CUdeviceptr = ctypes.c_uint64


class GPUError(RuntimeError):
    pass


@dataclass
class Buffer:
    """A device buffer created from host bytes; optionally read back after the run."""

    data: bytes
    read_back: bool = False


@dataclass
class Timing:
    median_ms: float
    q1_ms: float
    q3_ms: float
    samples: list[float] = field(repr=False)

    @property
    def iqr_ms(self) -> float:
        return self.q3_ms - self.q1_ms


@dataclass
class RunResult:
    timing: Timing
    outputs: dict[int, bytes]  # argument index -> bytes read back


class Driver:
    """A CUDA context on one device, with the driver-API calls we need."""

    def __init__(self, device: int = 0):
        name = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
        try:
            self.lib = ctypes.CDLL(name)
        except OSError as exc:
            raise GPUError(f"CUDA driver not found ({name}): {exc}") from exc
        self._check(self.lib.cuInit(0), "cuInit")
        self.device = ctypes.c_int()
        self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), device), "cuDeviceGet")
        # The primary context is the one the CUDA runtime would also use.
        self.ctx = ctypes.c_void_p()
        self._check(
            self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.device),
            "cuDevicePrimaryCtxRetain",
        )
        self._check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    # ------------------------------------------------------------------ info

    def _check(self, result: int, what: str) -> None:
        if result != 0:
            msg = ctypes.c_char_p()
            self.lib.cuGetErrorString(result, ctypes.byref(msg))
            text = msg.value.decode() if msg.value else f"error {result}"
            raise GPUError(f"{what}: {text}")

    def name(self) -> str:
        buf = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(buf, 256, self.device), "cuDeviceGetName")
        return buf.value.decode()

    def driver_version(self) -> str:
        v = ctypes.c_int()
        self._check(self.lib.cuDriverGetVersion(ctypes.byref(v)), "cuDriverGetVersion")
        return f"{v.value // 1000}.{(v.value % 1000) // 10}"

    def compute_capability(self) -> str:
        major, minor = ctypes.c_int(), ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75, _MINOR = 76
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(major), 75, self.device), "attr")
        self._check(self.lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, self.device), "attr")
        return f"sm_{major.value}{minor.value}"

    # ------------------------------------------------------------------- run

    def run(
        self,
        cubin: bytes,
        kernel: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: list,
        shared_bytes: int = 0,
        warmup: int = 5,
        repeats: int = 15,
        batch: int = 10,
    ) -> RunResult:
        """Launch ``kernel`` from ``cubin`` with ``args``, check-run it once,
        then time it.

        ``args`` holds Buffer objects (become device pointers) and ctypes
        scalars (``ctypes.c_int(n)``, ``ctypes.c_float(x)``) in parameter order.
        Buffers marked ``read_back`` are returned from the first (untimed) run.
        """
        lib = self.lib
        module, func = ctypes.c_void_p(), ctypes.c_void_p()
        self._check(lib.cuModuleLoadData(ctypes.byref(module), cubin), "cuModuleLoadData")
        allocations: dict[int, CUdeviceptr] = {}
        try:
            self._check(
                lib.cuModuleGetFunction(ctypes.byref(func), module, kernel.encode()),
                f"cuModuleGetFunction({kernel})",
            )
            values = []
            for i, arg in enumerate(args):
                if isinstance(arg, Buffer):
                    ptr = CUdeviceptr()
                    self._check(lib.cuMemAlloc_v2(ctypes.byref(ptr), len(arg.data)), "cuMemAlloc")
                    allocations[i] = ptr
                    values.append(ptr)
                else:
                    values.append(arg)
            # cuLaunchKernel takes an array of pointers to each argument value.
            params = (ctypes.c_void_p * len(values))(
                *(ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values)
            )

            def upload() -> None:
                for i, ptr in allocations.items():
                    data = args[i].data
                    self._check(lib.cuMemcpyHtoD_v2(ptr, data, len(data)), "cuMemcpyHtoD")

            def launch() -> None:
                self._check(
                    lib.cuLaunchKernel(func, *grid, *block, shared_bytes, None, params, None),
                    f"cuLaunchKernel({kernel})",
                )

            # One checked run for the outputs.
            upload()
            launch()
            self._check(lib.cuCtxSynchronize(), "cuCtxSynchronize")
            outputs = {}
            for i, ptr in allocations.items():
                if args[i].read_back:
                    host = ctypes.create_string_buffer(len(args[i].data))
                    self._check(lib.cuMemcpyDtoH_v2(host, ptr, len(args[i].data)), "cuMemcpyDtoH")
                    outputs[i] = host.raw

            # Timing. Each sample times `batch` back-to-back launches and divides,
            # which hides per-launch submission overhead (large under the
            # Windows WDDM driver model). Inputs are not re-uploaded: kernels
            # that accumulate (atomics) produce different values, not
            # different work, and correctness was checked on the run above.
            for _ in range(warmup):
                launch()
            start, stop = ctypes.c_void_p(), ctypes.c_void_p()
            self._check(lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
            self._check(lib.cuEventCreate(ctypes.byref(stop), 0), "cuEventCreate")
            samples = []
            for _ in range(repeats):
                self._check(lib.cuEventRecord(start, None), "cuEventRecord")
                for _ in range(batch):
                    launch()
                self._check(lib.cuEventRecord(stop, None), "cuEventRecord")
                self._check(lib.cuEventSynchronize(stop), "cuEventSynchronize")
                ms = ctypes.c_float()
                self._check(lib.cuEventElapsedTime(ctypes.byref(ms), start, stop), "elapsed")
                samples.append(ms.value / batch)
            lib.cuEventDestroy_v2(start)
            lib.cuEventDestroy_v2(stop)
        finally:
            for ptr in allocations.values():
                lib.cuMemFree_v2(ptr)
            lib.cuModuleUnload(module)

        q1, med, q3 = statistics.quantiles(samples, n=4) if len(samples) > 1 else samples * 3
        return RunResult(Timing(med, q1, q3, samples), outputs)


# --------------------------------------------------------------------------
# Measurement conditions
# --------------------------------------------------------------------------

_SMI_FIELDS = (
    "utilization.gpu",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "pstate",
    "clocks_throttle_reasons.active",
)


def gpu_state() -> dict:
    """Utilisation, temperature, clocks, throttle reasons and other GPU
    processes, read with ``nvidia-smi``. Recorded next to every measurement:
    timings taken while another application uses the GPU are not meaningful."""
    import shutil
    import subprocess

    smi = shutil.which("nvidia-smi")
    if smi is None:
        return {"available": False}
    query = [smi, f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"]
    apps = [smi, "--query-compute-apps=process_name", "--format=csv,noheader"]
    try:
        values = subprocess.run(query, capture_output=True, text=True, timeout=20).stdout
        names = subprocess.run(apps, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    first = values.strip().splitlines()[0] if values.strip() else ""
    state = dict(zip(_SMI_FIELDS, (v.strip() for v in first.split(",")), strict=False))
    state["available"] = True
    state["other_processes"] = sorted(
        {n.strip().replace("\\", "/").rsplit("/", 1)[-1] for n in names.splitlines() if n.strip()}
    )
    return state


def busy_reason(state: dict, max_utilization: int = 10) -> str | None:
    """Why the GPU should not be used for timing right now, or None if idle."""
    if not state.get("available"):
        return None
    try:
        util = int(state.get("utilization.gpu", "0"))
    except ValueError:
        util = 0
    others = [p for p in state.get("other_processes", []) if not p.lower().startswith("python")]
    if util > max_utilization or others:
        listed = ", ".join(others) or "unknown"
        return f"GPU is {util}% utilised by other processes ({listed})"
    return None
