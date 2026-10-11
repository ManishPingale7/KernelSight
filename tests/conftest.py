from functools import cache

import pytest

from kernelsight import toolchain


@cache
def _has_toolchain() -> bool:
    return toolchain.ptxas_status().available and toolchain.nvrtc_status().available


@cache
def _has_gpu() -> bool:
    from kernelsight.gpu import Driver, GPUError

    try:
        Driver()
    except GPUError:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    """Skip ``toolchain`` tests without NVRTC/ptxas and ``gpu`` tests without a
    CUDA-capable GPU (e.g. in CI)."""
    for item in items:
        if "toolchain" in item.keywords and not _has_toolchain():
            item.add_marker(pytest.mark.skip(reason="NVIDIA toolchain not installed"))
        elif "gpu" in item.keywords and not (_has_toolchain() and _has_gpu()):
            item.add_marker(pytest.mark.skip(reason="no CUDA GPU/driver (or toolchain) available"))
