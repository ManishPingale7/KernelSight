from pathlib import Path

import pytest

from kernelsight import toolchain

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``toolchain`` when NVRTC or ptxas is not installed."""
    if toolchain.ptxas_status().available and toolchain.nvrtc_status().available:
        return
    skip = pytest.mark.skip(reason="NVIDIA toolchain not installed (pip install -e .[toolchain])")
    for item in items:
        if "toolchain" in item.keywords:
            item.add_marker(skip)
