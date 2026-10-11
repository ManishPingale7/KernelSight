import json
from pathlib import Path

import pytest

from kernelsight import __version__, toolchain
from kernelsight.cli import main

PTX = Path(__file__).parent / "fixtures" / "ptx"
KERNELS = Path(__file__).parent.parent / "examples" / "kernels"


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


def test_analyze_json(capsys):
    assert main(["analyze", str(PTX / "reduction.sm89.ptx"), "--format", "json", "--no-ptxas"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert [f["name"] for f in rep["functions"]] == ["reduce_shared", "reduce_warp_shuffle"]


def test_analyze_writes_html_file(tmp_path, capsys):
    out = tmp_path / "report.html"
    args = [
        "analyze",
        str(PTX / "matmul.sm89.ptx"),
        "--format",
        "html",
        "--no-ptxas",
        "-o",
        str(out),
    ]
    assert main(args) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_compare_two_ptx_files(capsys):
    a, b = PTX / "elementwise.sm89.ptx", PTX / "elementwise.sm89_debug.ptx"
    assert main(["compare", str(a), str(b), "--no-ptxas"]) == 0
    out = capsys.readouterr().out
    assert "_Z11make_float4ffff`: only in b" in out


def test_cfg_dot(capsys):
    assert main(["cfg", str(PTX / "elementwise.sm89.ptx"), "-f", "saxpy"]) == 0
    dot = capsys.readouterr().out
    assert dot.startswith('digraph "saxpy"') and 'label="back"' in dot


def test_cfg_unknown_function(capsys):
    assert main(["cfg", str(PTX / "elementwise.sm89.ptx"), "-f", "nope"]) == 1
    assert "available: vector_add, saxpy, scale_vec4" in capsys.readouterr().err


def test_errors_exit_cleanly(tmp_path, capsys):
    bad = tmp_path / "bad.ptx"
    bad.write_text(".entry k() {\n ret\n}")
    assert main(["analyze", str(bad), "--no-ptxas"]) == 1
    assert "PTX syntax error at line 3" in capsys.readouterr().err
    assert main(["analyze", str(tmp_path / "missing.ptx")]) == 1


@pytest.mark.toolchain
def test_toolchain_versions():
    assert toolchain.ptxas_status().version.startswith("Cuda compilation tools")
    major = int(toolchain.nvrtc_status().version.split(".")[0])
    assert major >= 12


@pytest.mark.toolchain
def test_analyze_cuda_source_with_ptxas(capsys):
    args = ["analyze", str(KERNELS / "matmul.cu"), "--arch", "sm_75", "--format", "json"]
    assert main(args) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["provenance"]["nvrtc"]["options"] == ["-arch=compute_75"]
    assert rep["provenance"]["ptxas"]["arch"] == "sm_75"
    tiled = next(f for f in rep["functions"] if f["name"] == "matmul_tiled")
    assert tiled["ptxas"]["shared_mem"] == 2 * 16 * 16 * 4  # two 16x16 float tiles


@pytest.mark.toolchain
def test_compare_one_cuda_file_under_two_options(capsys):
    args = [
        "compare",
        str(KERNELS / "math_heavy.cu"),
        "--b-opt=--use_fast_math",
        "--format",
        "json",
    ]
    assert main(args) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["b"]["label"] == "math_heavy.cu --use_fast_math"
    assert "math_heavy" in rep["summary"]["changed"]
