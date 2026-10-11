"""Compare two compilations of the same code.

Typical use: compile one kernel twice with different options (e.g. with and
without ``--use_fast_math``, for sm_75 vs sm_89, with ``-G``) and ask *what
did the compiler do differently?*

Functions are matched by name. For each matched pair the comparison
reports changes in instruction mix, control flow, register pressure, ptxas
resources and findings, plus a textual diff of the instructions.

The textual diff is *normalised* first. Compilers number virtual registers
and labels sequentially, so one extra instruction early in a function
renumbers everything after it and a plain text diff becomes noise. We
rename registers (per register prefix) and labels in order of first
appearance, so ``%r17`` in both versions refers to "the 17th distinct %r
value used", and only real changes remain.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

from kernelsight.analysis import FunctionAnalysis, ModuleAnalysis
from kernelsight.ptx import ir

Pair = tuple[int, int]  # (value in A, value in B)


@dataclass
class FunctionDiff:
    name: str
    a: FunctionAnalysis | None
    b: FunctionAnalysis | None
    categories: dict[str, Pair] = field(default_factory=dict)
    memory: dict[str, Pair] = field(default_factory=dict)  # "load.global" -> (a, b)
    mnemonics: dict[str, Pair] = field(default_factory=dict)  # only those that changed
    metrics: dict[str, tuple[int | None, int | None]] = field(default_factory=dict)
    findings_added: list[str] = field(default_factory=list)
    findings_removed: list[str] = field(default_factory=list)
    diff: list[str] = field(default_factory=list)  # unified diff lines

    @property
    def status(self) -> str:
        if self.a is None:
            return "only_in_b"
        if self.b is None:
            return "only_in_a"
        return "changed" if self.diff else "identical"


@dataclass
class ModuleDiff:
    label_a: str
    label_b: str
    header: dict[str, tuple[str, str]]  # only fields that differ
    functions: list[FunctionDiff]


def compare(
    a: ModuleAnalysis, b: ModuleAnalysis, label_a: str = "A", label_b: str = "B"
) -> ModuleDiff:
    header = {}
    for key, va, vb in (
        ("version", a.module.version, b.module.version),
        ("target", ", ".join(a.module.target), ", ".join(b.module.target)),
        ("address_size", str(a.module.address_size), str(b.module.address_size)),
    ):
        if va != vb:
            header[key] = (va, vb)

    names_a = [f.name for f in a.functions]
    names_b = [f.name for f in b.functions]
    ordered = names_a + [n for n in names_b if n not in names_a]
    by_a = {f.name: f for f in a.functions}
    by_b = {f.name: f for f in b.functions}
    diffs = [compare_functions(by_a.get(n), by_b.get(n), n, label_a, label_b) for n in ordered]
    return ModuleDiff(label_a, label_b, header, diffs)


def compare_functions(
    a: FunctionAnalysis | None,
    b: FunctionAnalysis | None,
    name: str,
    label_a: str = "A",
    label_b: str = "B",
) -> FunctionDiff:
    fd = FunctionDiff(name, a, b)
    if a is None or b is None:
        return fd

    fd.categories = _pairs(a.mix.by_category, b.mix.by_category)
    fd.memory = _pairs(
        Counter({f"{k}.{s}": n for (k, s), n in a.mix.memory.items()}),
        Counter({f"{k}.{s}": n for (k, s), n in b.mix.memory.items()}),
    )
    mn_a = Counter(i.mnemonic for i in a.function.instructions)
    mn_b = Counter(i.mnemonic for i in b.function.instructions)
    fd.mnemonics = {k: v for k, v in _pairs(mn_a, mn_b).items() if v[0] != v[1]}

    fd.metrics = {
        "instructions": (a.mix.total, b.mix.total),
        "instructions_in_loops": (a.loop_mix.total, b.loop_mix.total),
        "basic_blocks": (len(a.cfg.blocks), len(b.cfg.blocks)),
        "loops": (len(a.cfg.loops), len(b.cfg.loops)),
        "virtual_registers": (a.registers.declared_count, b.registers.declared_count),
        "peak_live_slots": (a.registers.max_live_slots, b.registers.max_live_slots),
    }
    if a.resources is not None or b.resources is not None:
        ra, rb = a.resources, b.resources
        for key in ("registers", "spill_stores", "spill_loads", "stack_frame", "shared_mem"):
            fd.metrics[f"ptxas_{key}"] = (
                getattr(ra, key) if ra else None,
                getattr(rb, key) if rb else None,
            )

    rules_a = {f.rule for f in a.findings}
    rules_b = {f.rule for f in b.findings}
    fd.findings_added = sorted(rules_b - rules_a)
    fd.findings_removed = sorted(rules_a - rules_b)

    fd.diff = list(
        difflib.unified_diff(
            normalized_listing(a.function),
            normalized_listing(b.function),
            fromfile=f"{label_a}:{name}",
            tofile=f"{label_b}:{name}",
            lineterm="",
            n=2,
        )
    )
    return fd


def _pairs(a: Counter, b: Counter) -> dict[str, Pair]:
    return {k: (a.get(k, 0), b.get(k, 0)) for k in sorted(set(a) | set(b))}


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_REGISTER = re.compile(r"%([A-Za-z_]+)(\d+)\b")


def normalized_listing(fn: ir.Function) -> list[str]:
    """Instructions and labels of ``fn`` with registers and labels renumbered
    in order of first appearance. Special registers (``%tid.x``) and
    parameter names are left unchanged."""
    reg_ids: dict[str, int] = {}
    per_prefix: Counter[str] = Counter()
    label_ids: dict[str, str] = {}

    def reg(match: re.Match) -> str:
        full = match.group(0)
        if full not in reg_ids:
            per_prefix[match.group(1)] += 1
            reg_ids[full] = per_prefix[match.group(1)]
        return f"%{match.group(1)}{reg_ids[full]}"

    def label(name: str) -> str:
        if name not in label_ids:
            label_ids[name] = f"$L{len(label_ids) + 1}"
        return label_ids[name]

    lines = []
    for stmt in fn.body or []:
        if isinstance(stmt, ir.Label):
            lines.append(f"{label(stmt.name)}:")
        elif isinstance(stmt, ir.Instruction):
            text = _REGISTER.sub(reg, str(stmt))
            if stmt.opcode in ("bra", "brx") and stmt.operands:
                target = str(stmt.operands[0])
                text = text.replace(target, label(target))
            lines.append("    " + text)
    return lines
