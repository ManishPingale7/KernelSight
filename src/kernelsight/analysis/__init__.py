"""Static analyses over the PTX IR.

``analyze(module)`` runs everything for each function with a body:

* instruction mix (:mod:`.mix`), overall and for instructions inside loops
* control-flow graph, dominators and loops (:mod:`.cfg`)
* virtual-register def-use and liveness (:mod:`.registers`)
* ptxas resource usage, if ptxas output is supplied (:mod:`.resources`)
* rule-based findings (:mod:`.findings`)
"""

from __future__ import annotations

from dataclasses import dataclass

from kernelsight.analysis.cfg import CFG, build_cfg
from kernelsight.analysis.findings import Finding, find_issues
from kernelsight.analysis.mix import InstructionMix, instruction_mix
from kernelsight.analysis.registers import RegisterInfo, analyze_registers
from kernelsight.analysis.resources import ResourceUsage, parse_ptxas_output
from kernelsight.ptx import ir


@dataclass
class FunctionAnalysis:
    function: ir.Function
    mix: InstructionMix
    loop_mix: InstructionMix  # only instructions with loop depth > 0
    cfg: CFG
    registers: RegisterInfo
    resources: ResourceUsage | None
    findings: list[Finding]

    @property
    def name(self) -> str:
        return self.function.name


@dataclass
class ModuleAnalysis:
    module: ir.Module
    functions: list[FunctionAnalysis]
    ptxas_ran: bool

    def function(self, name: str) -> FunctionAnalysis:
        for fa in self.functions:
            if fa.name == name:
                return fa
        raise KeyError(name)


def analyze(module: ir.Module, ptxas_output: str | None = None) -> ModuleAnalysis:
    resources = parse_ptxas_output(ptxas_output) if ptxas_output else {}
    results = []
    for fn in module.functions:
        if fn.body is None:
            continue
        cfg = build_cfg(fn)
        mix = instruction_mix(fn.instructions)
        in_loops = [i for b in cfg.blocks if b.loop_depth > 0 for i in b.instructions]
        usage = resources.get(fn.name)
        results.append(
            FunctionAnalysis(
                function=fn,
                mix=mix,
                loop_mix=instruction_mix(in_loops),
                cfg=cfg,
                registers=analyze_registers(fn, cfg),
                resources=usage,
                findings=find_issues(fn, cfg, mix, usage),
            )
        )
    return ModuleAnalysis(module, results, ptxas_ran=ptxas_output is not None)


def target_arch(module: ir.Module) -> str | None:
    """The ``sm_XX`` target named in the module header, if any."""
    return next((t for t in module.target if t.startswith("sm_")), None)


__all__ = [
    "CFG",
    "Finding",
    "FunctionAnalysis",
    "InstructionMix",
    "ModuleAnalysis",
    "RegisterInfo",
    "ResourceUsage",
    "analyze",
    "target_arch",
]
