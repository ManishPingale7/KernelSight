"""Virtual-register analysis: declarations, def-use and liveness.

PTX registers are *virtual*: a kernel may declare thousands (``%f<500>``)
and the PTX assembler (ptxas) decides how many physical registers are
really used. That is why the number of declared registers says little about
register pressure, and why ptxas's own report (``ptxas -v``) is the ground
truth for the physical count.

What can be measured on PTX is *liveness*: at each instruction, which
values have been computed and are still needed later. The maximum number of
simultaneously live values is a lower-level signal of pressure. We count it
in 32-bit "slots" (a 64-bit register occupies two) and keep predicates
separate because they live in their own register file.

Liveness is the classic backward data-flow problem:

    live_out(B) = ⋃ live_in(S) for successors S
    live_in(B)  = use(B) ∪ (live_out(B) − def(B))

iterated until nothing changes. This estimate ignores what ptxas does
afterwards (instruction scheduling, rematerialisation, spilling), so treat
it as an explainable approximation and compare it with ptxas's report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelsight.analysis.cfg import CFG
from kernelsight.ptx import ir

# Instructions whose first operand is read, not written.
NO_DESTINATION = {
    "st", "bra", "brx", "ret", "exit", "bar", "barrier", "membar", "fence", "red",
    "prefetch", "prefetchu", "trap", "call", "cp", "mbarrier", "nanosleep",
}  # fmt: skip

_SLOTS = {"8": 1, "16": 1, "32": 1, "64": 2, "128": 4}


@dataclass
class RegisterInfo:
    declared: dict[str, str] = field(default_factory=dict)  # name -> type
    defined: set[str] = field(default_factory=set)
    used: set[str] = field(default_factory=set)
    max_live_slots: int = 0  # 32-bit register slots
    max_live_predicates: int = 0
    max_live_line: int | None = None  # source line where max_live_slots occurs

    @property
    def declared_count(self) -> int:
        return len(self.declared)

    @property
    def unused(self) -> set[str]:
        """Declared but never read or written (harmless; ptxas drops them)."""
        return set(self.declared) - self.defined - self.used


def slots(reg_type: str) -> int:
    """32-bit slots for a register type: f64 -> 2, f32 -> 1, pred -> 0."""
    if reg_type == "pred":
        return 0
    bits = "".join(ch for ch in reg_type if ch.isdigit())
    return _SLOTS.get(bits, 1)


def declared_registers(fn: ir.Function) -> dict[str, str]:
    """All register names declared in a function, mapped to their type.

    ``.reg .f32 %f<4>;`` declares %f0, %f1, %f2, %f3.
    """
    regs: dict[str, str] = {}
    for stmt in fn.body or []:
        if isinstance(stmt, ir.Declaration) and stmt.state_space == "reg":
            if stmt.count is not None:
                for i in range(stmt.count):
                    regs[f"{stmt.name}{i}"] = stmt.type or "?"
            else:
                for name in (stmt.name, *stmt.others):
                    regs[name] = stmt.type or "?"
    return regs


def registers_in(op: ir.Operand) -> list[str]:
    """Register names mentioned anywhere inside an operand."""
    if isinstance(op, ir.Register):
        return [op.name]
    if isinstance(op, ir.Address):
        return [op.base.name] if isinstance(op.base, ir.Register) else []
    if isinstance(op, (ir.Vector, ir.ParenList)):
        return [r for e in op.elements for r in registers_in(e)]
    if isinstance(op, ir.Negated):
        return registers_in(op.operand)
    if isinstance(op, ir.PredPair):
        return registers_in(op.first) + registers_in(op.second)
    return []


def defs_uses(inst: ir.Instruction, known: dict[str, str]) -> tuple[set[str], set[str]]:
    """Registers written and read by one instruction (special registers such
    as %tid.x are excluded because they are not in ``known``)."""
    defs: set[str] = set()
    uses: set[str] = set()
    operands = inst.operands
    if operands and inst.opcode not in NO_DESTINATION and not isinstance(operands[0], ir.Address):
        defs.update(registers_in(operands[0]))
        operands = operands[1:]
    for op in operands:
        uses.update(registers_in(op))
    if inst.predicate is not None:
        uses.add(inst.predicate.register.name)
    return defs & known.keys(), uses & known.keys()


def analyze_registers(fn: ir.Function, cfg: CFG) -> RegisterInfo:
    info = RegisterInfo(declared=declared_registers(fn))
    known = info.declared

    # Per-block use/def sets (use = read before any write in the block).
    block_use: dict[int, set[str]] = {}
    block_def: dict[int, set[str]] = {}
    for b in cfg.blocks:
        use: set[str] = set()
        dfn: set[str] = set()
        for inst in b.instructions:
            d, u = defs_uses(inst, known)
            info.defined |= d
            info.used |= u
            use |= u - dfn
            dfn |= d
        block_use[b.id], block_def[b.id] = use, dfn

    # Backward data-flow to a fixed point.
    live_in: dict[int, set[str]] = {b.id: set() for b in cfg.blocks}
    live_out: dict[int, set[str]] = {b.id: set() for b in cfg.blocks}
    changed = True
    while changed:
        changed = False
        for b in reversed(cfg.blocks):
            out = set().union(*(live_in[s] for s in b.successors))
            inn = block_use[b.id] | (out - block_def[b.id])
            if out != live_out[b.id] or inn != live_in[b.id]:
                live_out[b.id], live_in[b.id], changed = out, inn, True

    # Walk each block backwards to find the point of maximum pressure.
    for b in cfg.blocks:
        live = set(live_out[b.id])
        for inst in reversed(b.instructions):
            d, u = defs_uses(inst, known)
            # Pressure at an instruction = values live after it plus the
            # values it writes. Operands that die here are not added: the
            # result may reuse their registers.
            _record(info, live | d, inst.line, known)
            live = (live - d) | u
    return info


def _record(info: RegisterInfo, live: set[str], line: int, known: dict[str, str]) -> None:
    n_slots = sum(slots(known[r]) for r in live)
    n_preds = sum(1 for r in live if known[r] == "pred")
    if n_slots > info.max_live_slots:
        info.max_live_slots, info.max_live_line = n_slots, line
    info.max_live_predicates = max(info.max_live_predicates, n_preds)
