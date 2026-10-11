"""Static instruction mix of a function.

"Static" means each instruction is counted once, where it appears in the
code. An instruction inside a loop that runs 1000 times still counts as 1.
The CFG analysis (:mod:`kernelsight.analysis.cfg`) adds loop depth so a
reader can see which instructions are likely to execute repeatedly.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from kernelsight.analysis import opcodes
from kernelsight.ptx import ir


@dataclass
class InstructionMix:
    total: int = 0
    by_category: Counter[str] = field(default_factory=Counter)
    # (kind, space) -> count, e.g. ("load", "global") -> 4
    memory: Counter[tuple[str, str]] = field(default_factory=Counter)
    by_opcode: Counter[str] = field(default_factory=Counter)
    predicated: int = 0  # instructions guarded by @%p / @!%p
    uncategorized: list[str] = field(default_factory=list)  # opcodes in "other"

    def memory_ops(self, kind: str | None = None, space: str | None = None) -> int:
        return sum(
            n
            for (k, s), n in self.memory.items()
            if (kind is None or k == kind) and (space is None or s == space)
        )


def instruction_mix(instructions: list[ir.Instruction]) -> InstructionMix:
    mix = InstructionMix()
    for inst in instructions:
        mix.total += 1
        cat = opcodes.category(inst)
        mix.by_category[cat] += 1
        mix.by_opcode[inst.opcode] += 1
        if inst.predicate is not None:
            mix.predicated += 1
        if cat == "memory":
            mix.memory[(opcodes.memory_kind(inst), opcodes.memory_space(inst))] += 1
        elif cat == "other" and inst.opcode not in mix.uncategorized:
            mix.uncategorized.append(inst.opcode)
    return mix
