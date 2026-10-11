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
            # Skip the "builtins" helper library; we want the compiler itself.
            libs = sorted(p for p in folder.glob(pattern) if "builtins" not in p.name)
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
    # The last line looks like: "Cuda compilation tools, release 12.9, V12.9.86"
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return ToolStatus("ptxas", path, lines[-1].strip() if lines else None)


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
