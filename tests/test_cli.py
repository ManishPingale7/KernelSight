import pytest

from kernelsight import __version__, toolchain
from kernelsight.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_doctor_runs_without_toolchain(capsys, monkeypatch):
    monkeypatch.setenv("KERNELSIGHT_PTXAS", "/nonexistent/ptxas")
    monkeypatch.setenv("KERNELSIGHT_NVRTC", "/nonexistent/nvrtc.dll")
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ptxas" in out and "nvrtc" in out


@pytest.mark.toolchain
def test_toolchain_versions():
    assert toolchain.ptxas_status().version.startswith("Cuda compilation tools")
    major = int(toolchain.nvrtc_status().version.split(".")[0])
    assert major >= 12
