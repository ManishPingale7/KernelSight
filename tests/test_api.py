from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from kernelsight.api import app  # noqa: E402

PTX = Path(__file__).parent / "fixtures" / "ptx"
client = TestClient(app)


def ptx_source(name: str, **extra) -> dict:
    return {"source": (PTX / name).read_text(), "kind": "ptx", "name": name, **extra}


def test_health():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert set(body["toolchain"]) == {"nvrtc", "ptxas"}


def test_examples_list_cuda_files():
    names = [e["name"] for e in client.get("/api/examples").json()]
    assert "matmul.cu" in names and all(n.endswith(".cu") for n in names)


def test_analyze_ptx_json():
    r = client.post("/api/analyze", json=ptx_source("histogram.sm89.ptx", use_ptxas=False))
    assert r.status_code == 200
    rep = r.json()
    assert rep["schema"] == "kernelsight.analysis/1"
    assert rep["functions"][0]["loops"]["count"] == 3


@pytest.mark.parametrize("fmt, marker", [("md", "# KernelSight"), ("html", "<!doctype html>")])
def test_analyze_other_formats(fmt, marker):
    body = ptx_source("elementwise.sm89.ptx", use_ptxas=False)
    r = client.post(f"/api/analyze?format={fmt}", json=body)
    assert r.status_code == 200 and r.text.startswith(marker)


def test_compare():
    pair = {
        "a": ptx_source("math_heavy.sm89.ptx", use_ptxas=False),
        "b": ptx_source("math_heavy.sm89_fastmath.ptx", use_ptxas=False),
    }
    rep = client.post("/api/compare", json=pair).json()
    assert rep["summary"]["identical"] == ["polynomial"]


def test_syntax_error_is_422_with_line():
    r = client.post("/api/analyze", json={"source": ".entry k() {\n ret\n}", "use_ptxas": False})
    assert r.status_code == 422
    assert "line 3" in r.json()["detail"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("ptxas_options", ["--output-file=C:/Windows/evil"]),
        ("nvrtc_options", ["-I/etc"]),
        ("arch", "sm_89; rm -rf /"),
    ],
)
def test_unsafe_options_rejected(field, value):
    r = client.post("/api/analyze", json={"source": ".entry k() { ret; }", field: value})
    assert r.status_code == 422


@pytest.mark.toolchain
def test_analyze_cuda_with_options():
    src = {
        "source": 'extern "C" __global__ void k(float* x) { x[threadIdx.x] /= 3.0f; }',
        "kind": "cuda",
        "name": "k.cu",
        "nvrtc_options": ["--use_fast_math"],
    }
    rep = client.post("/api/analyze", json=src).json()
    assert rep["provenance"]["nvrtc"]["options"] == ["-arch=compute_89", "--use_fast_math"]
    assert rep["functions"][0]["ptxas"]["registers"] > 0
