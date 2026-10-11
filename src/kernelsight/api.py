"""HTTP API over the library (FastAPI), used by the web UI.

Run with ``kernelsight serve`` (needs ``pip install -e .[api]``).

Endpoints
---------
GET  /api/health              version and toolchain availability
GET  /api/examples            the example CUDA kernels shipped in examples/kernels
POST /api/analyze[?format=]   analyse PTX or CUDA source   -> report (json|md|html)
POST /api/compare[?format=]   compare two inputs           -> report (json|md|html)

The API is meant to run locally. It still validates input: sources are size
limited and compiler options must match an allowlist, because options are
passed to NVRTC and ptxas (an option such as ptxas ``--output-file`` would
otherwise let a request write files).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from kernelsight import __version__, pipeline, report, toolchain
from kernelsight.compare import compare
from kernelsight.ptx import PTXSyntaxError
from kernelsight.report import html, markdown

MAX_SOURCE_BYTES = 1_000_000
ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "kernels"
UI_BUILD = ROOT / "frontend" / "out"  # produced by `npm run build` (static export)

_BOOL = "(true|false)"
NVRTC_ALLOWED = re.compile(
    r"^(-arch=compute_\d{2,3}[a-z]?|--use_fast_math|-G|-lineinfo|-maxrregcount=\d{1,3}"
    rf"|--ftz={_BOOL}|--prec-div={_BOOL}|--prec-sqrt={_BOOL}|--fmad={_BOOL}"
    r"|--extra-device-vectorization|-std=c\+\+(11|14|17|20)"
    r"|-D[A-Za-z_]\w*(=[\w.]*)?)$"
)
PTXAS_ALLOWED = re.compile(r"^(-maxrregcount=\d{1,3}|-O[0-3])$")
ARCH = re.compile(r"^sm_\d{2,3}[a-z]?$")


class Source(BaseModel):
    source: str = Field(max_length=MAX_SOURCE_BYTES)
    kind: Literal["ptx", "cuda"] = "ptx"
    name: str = Field(default="input", max_length=200)
    arch: str | None = None
    nvrtc_options: list[str] = Field(default_factory=list, max_length=16)
    ptxas_options: list[str] = Field(default_factory=list, max_length=8)
    use_ptxas: bool | None = None  # None: use ptxas if installed

    @field_validator("arch")
    @classmethod
    def _arch(cls, v: str | None) -> str | None:
        if v is not None and not ARCH.match(v):
            raise ValueError("arch must look like sm_89")
        return v

    @field_validator("nvrtc_options")
    @classmethod
    def _nvrtc(cls, v: list[str]) -> list[str]:
        return _check(v, NVRTC_ALLOWED, "NVRTC")

    @field_validator("ptxas_options")
    @classmethod
    def _ptxas(cls, v: list[str]) -> list[str]:
        return _check(v, PTXAS_ALLOWED, "ptxas")


def _check(options: list[str], allowed: re.Pattern, tool: str) -> list[str]:
    bad = [o for o in options if not allowed.match(o)]
    if bad:
        raise ValueError(f"{tool} option(s) not allowed: {', '.join(bad)}")
    return options


class ComparePair(BaseModel):
    a: Source
    b: Source


Format = Literal["json", "md", "html"]

app = FastAPI(title="KernelSight API", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _run(src: Source) -> pipeline.AnalyzedInput:
    try:
        return pipeline.analyze_source(
            src.source,
            src.name,
            is_cuda=src.kind == "cuda",
            nvrtc_options=src.nvrtc_options,
            arch=src.arch,
            use_ptxas=src.use_ptxas,
            ptxas_options=src.ptxas_options,
        )
    except PTXSyntaxError as exc:
        raise HTTPException(422, f"PTX syntax error at {exc}") from exc
    except toolchain.ToolchainError as exc:
        raise HTTPException(422, str(exc)) from exc


def _respond(rep: dict, fmt: Format, render_md, render_html):
    if fmt == "md":
        return PlainTextResponse(render_md(rep), media_type="text/markdown")
    if fmt == "html":
        return HTMLResponse(render_html(rep))
    return rep


@app.get("/api/health")
def health() -> dict:
    tools = {}
    for status in (toolchain.nvrtc_status(), toolchain.ptxas_status()):
        tools[status.name] = {"available": status.available, "version": status.version}
    return {"status": "ok", "version": __version__, "toolchain": tools}


@app.get("/api/examples")
def examples() -> list[dict]:
    return [
        {"name": p.name, "source": p.read_text(encoding="utf-8")}
        for p in sorted(EXAMPLES.glob("*.cu"))
    ]


@app.post("/api/analyze")
def analyze_endpoint(src: Source, format: Format = "json"):
    result = _run(src)
    rep = report.analysis_report(result.analysis, result.provenance)
    return _respond(rep, format, markdown.analysis_markdown, html.analysis_html)


@app.post("/api/compare")
def compare_endpoint(pair: ComparePair, format: Format = "json"):
    ra, rb = _run(pair.a), _run(pair.b)
    label_a = _label(pair.a)
    label_b = _label(pair.b)
    if label_a == label_b:
        label_a, label_b = f"A: {label_a}", f"B: {label_b}"
    diff = compare(ra.analysis, rb.analysis, label_a, label_b)
    rep = report.comparison_report(diff, ra.provenance, rb.provenance)
    return _respond(rep, format, markdown.comparison_markdown, html.comparison_html)


def _label(src: Source) -> str:
    parts = [src.name, *([f"--arch={src.arch}"] if src.arch else []), *src.nvrtc_options]
    return " ".join(parts)


# Serve the built web UI (if present) at "/". Registered last so the /api
# routes above take precedence.
if UI_BUILD.is_dir():
    app.mount("/", StaticFiles(directory=UI_BUILD, html=True), name="ui")
