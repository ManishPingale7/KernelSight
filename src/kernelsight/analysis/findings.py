"""Rule-based findings: short, evidence-backed observations about a kernel.

Every finding names the PTX lines it is based on and states its caveat.
Rules only report what can be read from the code (and, when available,
from ptxas's resource report); they do not claim to predict run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelsight.analysis import opcodes
from kernelsight.analysis.cfg import CFG
from kernelsight.analysis.mix import InstructionMix
from kernelsight.analysis.resources import ResourceUsage
from kernelsight.ptx import ir


@dataclass
class Finding:
    rule: str  # stable identifier, e.g. "local-memory"
    severity: str  # "info" or "warning"
    title: str
    detail: str
    lines: list[int] = field(default_factory=list)  # PTX source lines (evidence)
    caveat: str | None = None


def find_issues(
    fn: ir.Function, cfg: CFG, mix: InstructionMix, resources: ResourceUsage | None
) -> list[Finding]:
    insts = fn.instructions
    loop_depth = {id(i): b.loop_depth for b in cfg.blocks for i in b.instructions}
    found: list[Finding] = []

    # -- local memory ------------------------------------------------------
    local = [i.line for i in insts if opcodes.memory_space(i) == "local"]
    if local:
        caveat = (
            "ptxas may still keep these values in registers; its stack-frame "
            "size shows whether local memory is really used."
        )
        if resources is not None:
            caveat = (
                f"ptxas reports a {resources.stack_frame}-byte stack frame for this "
                "function"
                + (", so these accesses were optimised away." if resources.stack_frame == 0
                   else ", so local memory is used in the final code.")
            )  # fmt: skip
        found.append(
            Finding(
                "local-memory",
                "info",
                f"{len(local)} local-memory access(es) in PTX",
                "Per-thread arrays indexed with run-time values, or too large for "
                "registers, are placed in local memory. Despite the name it resides "
                "in device memory (cached in L1/L2), so it is much slower than registers.",
                local,
                caveat,
            )
        )

    # -- spills (needs ptxas) ---------------------------------------------
    if resources is not None and resources.spills:
        found.append(
            Finding(
                "register-spills",
                "warning",
                f"ptxas spilled registers ({resources.spill_stores} B stored, "
                f"{resources.spill_loads} B loaded per thread)",
                f"With {resources.registers} registers per thread, ptxas could not keep all "
                "live values in registers and moved some to local memory. Reducing values "
                "live at the same time, or relaxing a register limit, removes spills.",
                caveat="Spills are decided after PTX, so they have no PTX line numbers.",
            )
        )

    # -- FP64 arithmetic ---------------------------------------------------
    fp64 = [
        i.line
        for i in insts
        if opcodes.category(i) in ("fp_arith", "special_fn") and "f64" in i.modifiers
    ]
    if fp64:
        found.append(
            Finding(
                "fp64-arithmetic",
                "info",
                f"{len(fp64)} double-precision (f64) arithmetic instruction(s)",
                "FP64 throughput varies widely between GPUs: per the CUDA C++ Programming "
                "Guide it is 1/64 of FP32 on compute capability 8.6/8.9 (GeForce RTX 30/40) "
                "but 1/2 on 8.0/9.0 (A100/H100).",
                fp64,
                "Only matters if the kernel is limited by arithmetic, not memory.",
            )
        )

    # -- IEEE-rounded f32 division / sqrt / reciprocal ---------------------
    precise = [
        i.line
        for i in insts
        if i.opcode in ("div", "sqrt", "rcp")
        and "f32" in i.modifiers
        and ("rn" in i.modifiers or "full" in i.modifiers)
    ]
    if precise:
        found.append(
            Finding(
                "precise-f32-math",
                "info",
                f"{len(precise)} IEEE-rounded f32 division/sqrt/reciprocal instruction(s)",
                "Correctly rounded single-precision div/sqrt/rcp compile to multi-instruction "
                "sequences. --use_fast_math (or intrinsics such as __fdividef) uses the "
                "approximate hardware instructions instead.",
                precise,
                "Fast math changes numerical results; only use it where the accuracy "
                "loss is acceptable.",
            )
        )

    # -- atomics inside loops ---------------------------------------------
    atomics_in_loops = [
        i.line for i in insts if opcodes.memory_kind(i) == "atomic" and loop_depth.get(id(i), 0)
    ]
    if atomics_in_loops:
        found.append(
            Finding(
                "atomics-in-loop",
                "info",
                f"{len(atomics_in_loops)} atomic operation(s) inside a loop",
                "Atomics to the same address from many threads are serialised. Inside a "
                "loop this can dominate run time; per-block privatisation in shared memory "
                "or warp-level reduction first reduces contention.",
                atomics_in_loops,
                "Contention depends on the addresses touched at run time, which static "
                "analysis cannot see.",
            )
        )

    # -- coverage of the analysis itself ----------------------------------
    if mix.uncategorized:
        found.append(
            Finding(
                "uncategorized-opcodes",
                "info",
                "Opcodes not in KernelSight's classification table",
                "Counted under 'other': " + ", ".join(sorted(mix.uncategorized)),
            )
        )
    if cfg.has_indirect_branch or cfg.unresolved_targets:
        found.append(
            Finding(
                "incomplete-cfg",
                "warning",
                "Control-flow graph may be incomplete",
                "Indirect branches (brx.idx) or unknown branch targets were found, so loop "
                "and liveness results may be missing edges.",
            )
        )
    return found
