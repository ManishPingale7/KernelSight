"""From an input file to an analysis, recording how it was produced.

An input is either PTX (``.ptx``) or CUDA C++ (``.cu``, compiled with NVRTC
first). Optionally the PTX is assembled with ``ptxas -v`` for real resource
usage. Every step is recorded in a *provenance* dict (input hash, compiler
options, tool versions) that goes into the report, so a reader can
reproduce the result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from kernelsight import toolchain
from kernelsight.analysis import ModuleAnalysis, analyze, target_arch
from kernelsight.ptx import parse

DEFAULT_ARCH = "sm_89"


@dataclass
class AnalyzedInput:
    analysis: ModuleAnalysis
    provenance: dict
    ptx: str


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def analyze_source(
    text: str,
    name: str,
    *,
    is_cuda: bool = False,
    nvrtc_options: list[str] | None = None,
    arch: str | None = None,
    use_ptxas: bool | None = None,
    ptxas_options: list[str] | None = None,
) -> AnalyzedInput:
    """Analyse PTX or CUDA source text.

    ``arch`` is the ``sm_XX`` target. For CUDA input it selects NVRTC's
    ``-arch=compute_XX`` (unless ``nvrtc_options`` already sets ``-arch``);
    for ptxas it defaults to the PTX module's own ``.target``.
    ``use_ptxas=None`` means "use ptxas if it is installed".
    """
    provenance: dict = {"input": {"name": name, "sha256": sha256(text)}}

    if is_cuda:
        options = list(nvrtc_options or [])
        if not any(o.startswith(("-arch", "--gpu-architecture")) for o in options):
            options.insert(0, f"-arch=compute_{(arch or DEFAULT_ARCH).removeprefix('sm_')}")
        compiled = toolchain.compile_cuda(text, options, name=name)
        ptx = compiled.ptx
        provenance["input"]["kind"] = "cuda"
        provenance["nvrtc"] = {
            "version": compiled.nvrtc_version,
            "options": options,
            "log": compiled.log or None,
        }
    else:
        ptx = text
        provenance["input"]["kind"] = "ptx"

    module = parse(ptx)
    provenance["ptx_sha256"] = sha256(ptx)

    if use_ptxas is None:
        use_ptxas = toolchain.ptxas_status().available
    ptxas_output = None
    if use_ptxas:
        ptxas_arch = arch or target_arch(module) or DEFAULT_ARCH
        ptxas_output = toolchain.run_ptxas(ptx, ptxas_arch, ptxas_options or [])
        provenance["ptxas"] = {
            "version": toolchain.ptxas_status().version,
            "arch": ptxas_arch,
            "options": list(ptxas_options or []),
        }
    return AnalyzedInput(analyze(module, ptxas_output), provenance, ptx)


def analyze_file(path: str | Path, **kwargs) -> AnalyzedInput:
    path = Path(path)
    return analyze_source(
        path.read_text(encoding="utf-8"),
        path.name,
        is_cuda=path.suffix in (".cu", ".cuh"),
        **kwargs,
    )
