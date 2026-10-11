"""Locate the optional NVIDIA toolchain: NVRTC and ptxas.

KernelSight's parser and analyses work on PTX text alone. Two NVIDIA tools add
capabilities on top of that:

* NVRTC   - the CUDA runtime compiler library. It compiles CUDA C++ source to
            PTX without needing nvcc or a host C++ compiler.
* ptxas   - the PTX assembler. It turns PTX into GPU machine code (SASS) and,
            with ``-v``, reports the *real* register count, spills and shared
            memory chosen for each kernel.

Both ship as NVIDIA pip wheels (``pip install kernelsight[toolchain]``) and as
part of the CUDA Toolkit. Search order for each tool:

1. an explicit environment variable (KERNELSIGHT_PTXAS / KERNELSIGHT_NVRTC)
2. the NVIDIA pip wheels installed in this Python environment
3. the CUDA Toolkit pointed to by CUDA_PATH / CUDA_HOME
4. PATH (ptxas only)
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"
_EXE = ".exe" if _IS_WINDOWS else ""


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: Path | None
    version: str | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.path is not None and self.error is None


def _site_dirs() -> list[Path]:
    """Directories where pip installs packages for this interpreter."""
    import site

    dirs = [Path(p) for p in site.getsitepackages()]
    user = site.getusersitepackages()
    if user:
        dirs.append(Path(user))
    return dirs


def _toolkit_dirs() -> list[Path]:
    return [Path(os.environ[v]) for v in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(v)]


def find_ptxas() -> Path | None:
    override = os.environ.get("KERNELSIGHT_PTXAS")
    if override:
        return Path(override)
    candidates = [d / "nvidia" / "cuda_nvcc" / "bin" / f"ptxas{_EXE}" for d in _site_dirs()]
    candidates += [d / "bin" / f"ptxas{_EXE}" for d in _toolkit_dirs()]
    for c in candidates:
        if c.is_file():
            return c
    on_path = shutil.which("ptxas")
    return Path(on_path) if on_path else None


def find_nvrtc() -> Path | None:
    override = os.environ.get("KERNELSIGHT_NVRTC")
    if override:
        return Path(override)
    # Windows: bin/nvrtc64_120_0.dll; Linux: lib/libnvrtc.so.12
    pattern, sub = ("nvrtc64_*.dll", "bin") if _IS_WINDOWS else ("libnvrtc.so*", "lib")
    roots = [d / "nvidia" / "cuda_nvrtc" for d in _site_dirs()] + _toolkit_dirs()
    for root in roots:
        folder = root / sub
        if folder.is_dir():
            # Skip the "builtins" helper library and the ".alt" variant; we
            # want the main compiler library (e.g. nvrtc64_120_0.dll).
            libs = sorted(
                p for p in folder.glob(pattern) if "builtins" not in p.name and ".alt" not in p.name
            )
            if libs:
                return libs[0]
    return None


def ptxas_status() -> ToolStatus:
    path = find_ptxas()
    if path is None:
        return ToolStatus("ptxas", None, None, "not found")
    try:
        out = subprocess.run(
            [str(path), "--version"], capture_output=True, text=True, timeout=30, check=True
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return ToolStatus("ptxas", path, None, str(exc))
    # Output ends with two lines like:
    #   Cuda compilation tools, release 12.9, V12.9.86
    #   Build cuda_12.9.r12.9/compiler.36037853_0
    release = [ln.strip() for ln in out.splitlines() if "release" in ln]
    return ToolStatus("ptxas", path, release[-1] if release else out.strip() or None)


def nvrtc_status() -> ToolStatus:
    path = find_nvrtc()
    if path is None:
        return ToolStatus("nvrtc", None, None, "not found")
    try:
        lib = ctypes.CDLL(str(path))
        major, minor = ctypes.c_int(), ctypes.c_int()
        # nvrtcResult nvrtcVersion(int *major, int *minor); 0 == NVRTC_SUCCESS
        if lib.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)) != 0:
            return ToolStatus("nvrtc", path, None, "nvrtcVersion failed")
    except OSError as exc:
        return ToolStatus("nvrtc", path, None, str(exc))
    return ToolStatus("nvrtc", path, f"{major.value}.{minor.value}")


# --------------------------------------------------------------------------
# Running the tools
# --------------------------------------------------------------------------


class ToolchainError(RuntimeError):
    """A tool is missing, or it rejected its input (the log is in the message)."""


@dataclass(frozen=True)
class CompileResult:
    ptx: str
    log: str  # compiler warnings (may be empty)
    options: tuple[str, ...]
    nvrtc_version: str


_nvrtc_lib: ctypes.CDLL | None = None


def _load_nvrtc() -> ctypes.CDLL:
    global _nvrtc_lib
    if _nvrtc_lib is None:
        path = find_nvrtc()
        if path is None:
            raise ToolchainError("NVRTC not found; install with: pip install -e .[toolchain]")
        if _IS_WINDOWS:
            # NVRTC loads nvrtc-builtins64_*.dll at runtime with a plain
            # LoadLibrary call, which searches PATH (not add_dll_directory).
            os.environ["PATH"] = str(path.parent) + os.pathsep + os.environ.get("PATH", "")
        lib = ctypes.CDLL(str(path))
        lib.nvrtcGetErrorString.restype = ctypes.c_char_p
        _nvrtc_lib = lib
    return _nvrtc_lib


def compile_cuda(source: str, options: list[str] | tuple[str, ...] = (), name: str = "kernel.cu"):
    """Compile CUDA C++ source to PTX with NVRTC.

    ``options`` are NVRTC flags, e.g. ``["-arch=compute_89", "--use_fast_math"]``.
    NVRTC compiles device code only and needs no host compiler. Kernels that
    should be easy to find in the PTX can be declared ``extern "C"`` to avoid
    C++ name mangling.
    """
    lib = _load_nvrtc()

    def check(result: int, what: str) -> None:
        if result != 0:
            raise ToolchainError(f"{what}: {lib.nvrtcGetErrorString(result).decode()}")

    prog = ctypes.c_void_p()
    check(
        lib.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), name.encode(), 0, None, None),
        "nvrtcCreateProgram",
    )
    try:
        opts = (ctypes.c_char_p * len(options))(*(o.encode() for o in options))
        compile_result = lib.nvrtcCompileProgram(prog, len(options), opts)

        log_size = ctypes.c_size_t()
        check(lib.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size)), "nvrtcGetProgramLogSize")
        log_buf = ctypes.create_string_buffer(log_size.value)
        check(lib.nvrtcGetProgramLog(prog, log_buf), "nvrtcGetProgramLog")
        log = log_buf.value.decode(errors="replace").strip()
        if compile_result != 0:
            raise ToolchainError(f"NVRTC failed to compile {name}:\n{log}")

        ptx_size = ctypes.c_size_t()
        check(lib.nvrtcGetPTXSize(prog, ctypes.byref(ptx_size)), "nvrtcGetPTXSize")
        ptx_buf = ctypes.create_string_buffer(ptx_size.value)
        check(lib.nvrtcGetPTX(prog, ptx_buf), "nvrtcGetPTX")
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))

    status = nvrtc_status()
    return CompileResult(ptx_buf.value.decode(), log, tuple(options), status.version or "?")


def run_ptxas(ptx: str, arch: str = "sm_89", extra_args: list[str] | tuple[str, ...] = ()):
    """Assemble PTX with ``ptxas -v`` and return its combined output.

    ``-v`` makes ptxas print per-function resource usage (registers, spills,
    shared memory). Raises ToolchainError if the PTX is rejected.
    """
    return assemble(ptx, arch, extra_args)[0]


def assemble(
    ptx: str, arch: str = "sm_89", extra_args: list[str] | tuple[str, ...] = ()
) -> tuple[str, bytes]:
    """Run ``ptxas -v`` and return (resource report text, cubin bytes).

    The cubin is the GPU machine code ptxas produced; loading exactly this
    cubin for benchmarks ensures the code that is timed is the code whose
    resource usage was reported.
    """
    import tempfile

    path = find_ptxas()
    if path is None:
        raise ToolchainError("ptxas not found; install with: pip install -e .[toolchain]")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "input.ptx"
        out = Path(tmp) / "out.cubin"
        src.write_text(ptx, encoding="utf-8", newline="\n")
        cmd = [str(path), f"-arch={arch}", "-v", *extra_args, str(src), "-o", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        cubin = out.read_bytes() if proc.returncode == 0 else b""
    output = (proc.stdout + proc.stderr).replace(str(src), "input.ptx")
    if proc.returncode != 0:
        raise ToolchainError(f"ptxas rejected the PTX:\n{output.strip()}")
    return output, cubin
