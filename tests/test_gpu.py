"""GPU tests: run every benchmark workload once and check its output.

These check *correctness* (GPU result vs a Python reference), which is
valid even when the GPU is shared; they do not assert anything about timing.
"""

import pytest

from kernelsight.gpu import busy_reason


def test_busy_reason():
    assert busy_reason({"available": False}) is None
    idle = {"available": True, "utilization.gpu": "0", "other_processes": []}
    assert busy_reason(idle) is None
    gaming = {"available": True, "utilization.gpu": "99", "other_processes": ["Game.exe"]}
    assert "99%" in busy_reason(gaming) and "Game.exe" in busy_reason(gaming)
    # Our own Python process may show up as a GPU client; that is not "busy".
    assert busy_reason({**idle, "other_processes": ["python.exe"]}) is None


@pytest.fixture(scope="module")
def driver():
    from kernelsight.gpu import Driver

    return Driver()


def _workload_names():
    from kernelsight.bench import W

    return list(W)


@pytest.mark.gpu
@pytest.mark.parametrize("name", _workload_names())
def test_workload_output_is_correct(driver, name):
    from kernelsight.bench import W, run_workload

    entry = run_workload(driver, W[name], "sm_89", repeats=1)
    assert entry["correct"]
    assert entry["median_ms"] > 0


@pytest.mark.gpu
def test_spill_variant_really_spills(driver):
    from kernelsight.bench import W, static_metrics

    base, _ = static_metrics(W["many_live_values"], "sm_89")
    limited, _ = static_metrics(W["many_live_values maxrreg=24"], "sm_89")
    assert base["spill_bytes"] == 0
    assert limited["ptxas_registers"] <= 24 and limited["spill_bytes"] > 0
    assert "register-spills" in limited["findings"]
