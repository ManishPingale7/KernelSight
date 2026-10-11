"""Benchmark studies: do the static findings match measured behaviour?

Each *study* compares two variants of a kernel (two kernels, or one kernel
under two compiler options). For each variant we record:

* the static analysis (instruction counts, ptxas registers/spills, ...)
  of exactly the cubin that is run,
* a correctness check of the GPU output against a Python reference,
* the measured time (median and IQR over repeated batches),
* the GPU state (utilisation, clocks, throttling) during measurement.

This is an *evidence study*, not a predictive model: with a handful of
kernels on one laptop GPU there is far too little data to train or validate
a model, and the results are presented as observations with caveats.
"""

from __future__ import annotations

import ctypes
import math
import random
from array import array
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from kernelsight import pipeline, toolchain
from kernelsight.gpu import Buffer, Driver

KERNELS = Path(__file__).resolve().parents[2] / "examples" / "kernels"


@dataclass
class Workload:
    name: str
    source: str  # file in examples/kernels
    kernel: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    make_args: Callable[[], list]
    check: Callable[[list, dict[int, bytes]], None]  # (args, outputs); raises on mismatch
    bytes_moved: int | None = None  # minimum DRAM traffic, for effective bandwidth
    shared_bytes: int = 0
    nvrtc_options: list[str] = field(default_factory=list)
    ptxas_options: list[str] = field(default_factory=list)


def _floats(out: bytes) -> array:
    a = array("f")
    a.frombytes(out)
    return a


def _pattern(n: int, values=(0.0, 1.0, 2.0, 3.0)) -> array:
    """Small integers: float sums of them stay exact, so checks can use ==."""
    return array("f", values) * (n // len(values))


def _samples(n: int, k: int = 64) -> list[int]:
    rng = random.Random(0)
    return sorted({0, n - 1, *(rng.randrange(n) for _ in range(k))})


# --- element-wise -------------------------------------------------------------

N_ELEM = 1 << 24


def _vector_add_args():
    a, b = _pattern(N_ELEM), _pattern(N_ELEM, (5.0, 6.0, 7.0, 8.0))
    return [Buffer(a.tobytes()), Buffer(b.tobytes()), Buffer(bytes(4 * N_ELEM), True),
            ctypes.c_int(N_ELEM)]  # fmt: skip


def _vector_add_check(args, out):
    a, b, c = _floats(args[0].data), _floats(args[1].data), _floats(out[2])
    assert all(c[i] == a[i] + b[i] for i in _samples(N_ELEM)), "vector_add mismatch"


def _saxpy_args():
    x, y = _pattern(N_ELEM), _pattern(N_ELEM, (1.0, 1.0, 2.0, 2.0))
    return [ctypes.c_float(2.0), Buffer(x.tobytes()), Buffer(y.tobytes(), True),
            ctypes.c_int(N_ELEM)]  # fmt: skip


def _saxpy_check(args, out):
    x, y0, y = _floats(args[1].data), _floats(args[2].data), _floats(out[2])
    assert all(y[i] == 2.0 * x[i] + y0[i] for i in _samples(N_ELEM)), "saxpy mismatch"


# --- matmul -------------------------------------------------------------------

N_MAT = 1024


def _matmul_args():
    a = array("f", ((i * 7 + 3) % 4 for i in range(N_MAT * N_MAT)))
    b = array("f", ((i * 5 + 1) % 4 for i in range(N_MAT * N_MAT)))
    return [Buffer(a.tobytes()), Buffer(b.tobytes()), Buffer(bytes(4 * N_MAT * N_MAT), True),
            ctypes.c_int(N_MAT)]  # fmt: skip


def _matmul_check(args, out):
    a, b, c = _floats(args[0].data), _floats(args[1].data), _floats(out[2])
    for idx in _samples(N_MAT * N_MAT, 16):
        r, col = divmod(idx, N_MAT)
        expect = sum(a[r * N_MAT + k] * b[k * N_MAT + col] for k in range(N_MAT))
        assert c[idx] == expect, f"matmul mismatch at ({r}, {col})"


# --- transpose ----------------------------------------------------------------

N_T = 4096


def _transpose_args():
    src = array("f", range(N_T)) * N_T  # in[y*W + x] = x
    return [Buffer(src.tobytes()), Buffer(bytes(4 * N_T * N_T), True),
            ctypes.c_int(N_T), ctypes.c_int(N_T)]  # fmt: skip


def _transpose_check(args, out):
    src, dst = _floats(args[0].data), _floats(out[1])
    for idx in _samples(N_T * N_T):
        y, x = divmod(idx, N_T)
        assert dst[x * N_T + y] == src[y * N_T + x], "transpose mismatch"


# --- reductions ---------------------------------------------------------------

N_RED = 1 << 24  # sum of N_RED ones is exactly representable in float32


def _reduce_args():
    return [Buffer((array("f", [1.0]) * N_RED).tobytes()), Buffer(bytes(4), True),
            ctypes.c_int(N_RED)]  # fmt: skip


def _reduce_check(args, out):
    assert _floats(out[1])[0] == float(N_RED), f"sum is {_floats(out[1])[0]}, expected {N_RED}"


# --- math_heavy ---------------------------------------------------------------

N_MATH = 1 << 22


def _math_args():
    x = array("f", (4.0 * i / N_MATH for i in range(N_MATH)))
    return [Buffer(x.tobytes()), Buffer(bytes(4 * N_MATH), True), ctypes.c_int(N_MATH)]


def _math_reference(x: float) -> float:
    y = math.sqrt(x * x + 1.0) / (x + 2.0) + math.exp(-x) * math.sin(x) + math.log(1.0 + x * x)
    return y / (1.0 + abs(y))


def _math_check(tolerance: float):
    def check(args, out):
        x, y = _floats(args[0].data), _floats(out[1])
        worst = max(abs(y[i] - _math_reference(x[i])) for i in _samples(N_MATH, 512))
        assert worst <= tolerance, f"max abs error {worst:.2e} > {tolerance:.0e}"

    return check


# --- register pressure --------------------------------------------------------

N_REG = 1 << 22


def _live_args():
    return [Buffer(_pattern(N_REG).tobytes()), Buffer(bytes(4 * N_REG), True),
            ctypes.c_int(N_REG)]  # fmt: skip


def _live_check(args, out):
    x, y = _floats(args[0].data), _floats(out[1])
    for i in _samples(N_REG, 16):
        acc = [x[(i + k * 97) % N_REG] for k in range(32)]
        total = sum(acc[k] * acc[(k + r + 1) % 32] for r in range(4) for k in range(32))
        assert y[i] == total, "many_live_values mismatch"


def _grid1d(n: int, block: int = 256) -> tuple[int, int, int]:
    return ((n + block - 1) // block, 1, 1)


B256 = (256, 1, 1)
W: dict[str, Workload] = {
    w.name: w
    for w in [
        Workload("vector_add", "elementwise.cu", "vector_add", _grid1d(N_ELEM), B256,
                 _vector_add_args, _vector_add_check, bytes_moved=12 * N_ELEM),
        Workload("saxpy", "elementwise.cu", "saxpy", _grid1d(N_ELEM), B256,
                 _saxpy_args, _saxpy_check, bytes_moved=12 * N_ELEM),
        Workload("saxpy -G", "elementwise.cu", "saxpy", _grid1d(N_ELEM), B256,
                 _saxpy_args, _saxpy_check, bytes_moved=12 * N_ELEM, nvrtc_options=["-G"]),
        Workload("matmul_naive", "matmul.cu", "matmul_naive", (N_MAT // 16, N_MAT // 16, 1),
                 (16, 16, 1), _matmul_args, _matmul_check),
        Workload("matmul_tiled", "matmul.cu", "matmul_tiled", (N_MAT // 16, N_MAT // 16, 1),
                 (16, 16, 1), _matmul_args, _matmul_check),
        Workload("transpose_naive", "transpose.cu", "transpose_naive", (N_T // 32, N_T // 32, 1),
                 (32, 32, 1), _transpose_args, _transpose_check, bytes_moved=8 * N_T * N_T),
        Workload("transpose_tiled", "transpose.cu", "transpose_tiled", (N_T // 32, N_T // 32, 1),
                 (32, 32, 1), _transpose_args, _transpose_check, bytes_moved=8 * N_T * N_T),
        Workload("reduce_shared", "reduction.cu", "reduce_shared", _grid1d(N_RED), B256,
                 _reduce_args, _reduce_check, bytes_moved=4 * N_RED, shared_bytes=256 * 4),
        Workload("reduce_warp_shuffle", "reduction.cu", "reduce_warp_shuffle", _grid1d(N_RED),
                 B256, _reduce_args, _reduce_check, bytes_moved=4 * N_RED),
        Workload("math_heavy", "math_heavy.cu", "math_heavy", _grid1d(N_MATH), B256,
                 _math_args, _math_check(1e-5)),
        Workload("math_heavy --use_fast_math", "math_heavy.cu", "math_heavy", _grid1d(N_MATH),
                 B256, _math_args, _math_check(1e-3), nvrtc_options=["--use_fast_math"]),
        Workload("many_live_values", "register_pressure.cu", "many_live_values",
                 _grid1d(N_REG), B256, _live_args, _live_check),
        Workload("many_live_values maxrreg=24", "register_pressure.cu", "many_live_values",
                 _grid1d(N_REG), B256, _live_args, _live_check, ptxas_options=["-maxrregcount=24"]),
    ]
}  # fmt: skip

STUDIES: list[tuple[str, str, str, str]] = [
    # (title, workload A, workload B, question)
    ("Fast math", "math_heavy", "math_heavy --use_fast_math",
     "Does the large drop in instruction count translate into speed?"),
    ("Shared-memory tiling", "matmul_naive", "matmul_tiled",
     "Tiling adds instructions; does reusing data from shared memory win anyway?"),
    ("Transpose with a shared tile", "transpose_naive", "transpose_tiled",
     "Coalescing both reads and writes vs the naive strided writes"),
    ("Warp shuffle vs shared-memory reduction", "reduce_shared", "reduce_warp_shuffle",
     "Fewer barriers and no shared memory vs a block-level tree"),
    ("Forced register spills", "many_live_values", "many_live_values maxrreg=24",
     "What do the spills ptxas reports cost?"),
    ("Debug build (-G)", "saxpy", "saxpy -G",
     "Cost of disabling device optimisations on a memory-bound kernel"),
]  # fmt: skip


def static_metrics(w: Workload, arch: str) -> tuple[dict, bytes]:
    """Static analysis of the exact cubin that will be run."""
    source = (KERNELS / w.source).read_text(encoding="utf-8")
    result = pipeline.analyze_source(
        source, w.source, is_cuda=True, nvrtc_options=w.nvrtc_options, arch=arch,
        use_ptxas=True, ptxas_options=w.ptxas_options,
    )  # fmt: skip
    _, cubin = toolchain.assemble(result.ptx, arch, w.ptxas_options)
    fa = result.analysis.function(w.kernel)
    r = fa.resources
    metrics = {
        "instructions": fa.mix.total,
        "instructions_in_loops": fa.loop_mix.total,
        "global_memory_ops": fa.mix.memory_ops(space="global"),
        "shared_memory_ops": fa.mix.memory_ops(space="shared"),
        "ptxas_registers": r.registers if r else None,
        "spill_bytes": (r.spill_stores + r.spill_loads) if r else None,
        "findings": [f.rule for f in fa.findings],
    }
    return metrics, cubin


def run_workload(driver: Driver, w: Workload, arch: str, repeats: int = 15) -> dict:
    metrics, cubin = static_metrics(w, arch)
    args = w.make_args()
    result = driver.run(cubin, w.kernel, w.grid, w.block, args, w.shared_bytes, repeats=repeats)
    w.check(args, result.outputs)  # raises AssertionError on wrong output
    t = result.timing
    entry = {
        "workload": w.name,
        "kernel": w.kernel,
        "source": f"examples/kernels/{w.source}",
        "nvrtc_options": w.nvrtc_options,
        "ptxas_options": w.ptxas_options,
        "grid": list(w.grid),
        "block": list(w.block),
        "static": metrics,
        "correct": True,
        "median_ms": round(t.median_ms, 4),
        "iqr_ms": round(t.iqr_ms, 4),
    }
    if w.bytes_moved:
        entry["effective_gb_per_s"] = round(w.bytes_moved / (t.median_ms * 1e6), 1)
    return entry
