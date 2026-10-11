"""Classify PTX instructions into categories.

The table below groups opcodes the way a performance engineer reads PTX.
It is deliberately a plain table so it is easy to audit against the PTX ISA
manual (chapter 9, "Instruction Set"). Opcodes not in the table are
reported as ``other`` and listed explicitly, so gaps are never hidden.

Categories
----------
memory         ld, st, atom, red, ... (further split by state space)
fp_arith       floating-point add/mul/fma/... (type modifier f16/bf16/f32/f64)
int_arith      integer add/mul/mad/div/rem/...
special_fn     transcendental and approximate functions run on the SFU
               (sin, cos, ex2, lg2, rcp, rsqrt, sqrt, tanh)
logic          bitwise and shift operations
compare        setp/selp/set/slct (predicate computation and selection)
move_convert   mov, cvt, cvta (no arithmetic, but still issue slots)
control        bra, call, ret, exit
sync           barriers and memory fences
warp           warp-level data exchange (shfl, vote, match, redux)
tensor         matrix-multiply-accumulate (mma, wmma, ldmatrix)
other          anything not listed above
"""

from __future__ import annotations

from kernelsight.ptx import ir

MEMORY_OPS = {"ld", "ldu", "st", "atom", "red", "prefetch", "prefetchu", "cp"}
ARITH_OPS = {
    "add", "sub", "mul", "mad", "fma", "div", "rem", "abs", "neg", "min", "max",
    "mul24", "mad24", "sad", "addc", "subc", "madc", "copysign", "testp",
}  # fmt: skip
SPECIAL_FN_OPS = {"sin", "cos", "ex2", "lg2", "rcp", "rsqrt", "sqrt", "tanh"}
LOGIC_OPS = {
    "and", "or", "xor", "not", "cnot", "shl", "shr", "shf", "lop3", "bfe", "bfi",
    "bfind", "brev", "clz", "popc", "prmt", "fns", "bmsk", "szext",
}  # fmt: skip
COMPARE_OPS = {"setp", "selp", "set", "slct"}
MOVE_OPS = {"mov", "cvt", "cvta", "isspacep"}
CONTROL_OPS = {"bra", "brx", "call", "ret", "exit", "trap", "brkpt", "nanosleep"}
SYNC_OPS = {"bar", "barrier", "membar", "fence", "mbarrier", "griddepcontrol"}
WARP_OPS = {"shfl", "vote", "match", "redux", "activemask", "elect"}
TENSOR_OPS = {"mma", "wmma", "ldmatrix", "wgmma"}

FLOAT_TYPES = {"f16", "f16x2", "bf16", "bf16x2", "f32", "f64", "tf32"}

# Memory state spaces (PTX ISA 5.1). An ld/st without one uses a *generic*
# address that the hardware resolves at run time.
STATE_SPACES = ("global", "shared", "local", "const", "param", "tex")


def category(inst: ir.Instruction) -> str:
    op = inst.opcode
    if op in MEMORY_OPS:
        return "memory"
    if op in ARITH_OPS:
        return "fp_arith" if data_type(inst) in FLOAT_TYPES else "int_arith"
    if op in SPECIAL_FN_OPS:
        return "special_fn"
    if op in LOGIC_OPS:
        return "logic"
    if op in COMPARE_OPS:
        return "compare"
    if op in MOVE_OPS:
        return "move_convert"
    if op in CONTROL_OPS:
        return "control"
    if op in SYNC_OPS:
        return "sync"
    if op in WARP_OPS:
        return "warp"
    if op in TENSOR_OPS:
        return "tensor"
    return "other"


def data_type(inst: ir.Instruction) -> str | None:
    """The instruction's operand type: the last type-like modifier.

    ``fma.rn.f32`` -> ``f32``; ``ld.global.v4.f32`` -> ``f32``;
    ``mul.wide.s32`` -> ``s32``. For ``cvt.rn.f32.s32`` this returns the
    *source* type ``s32``; use :func:`types` for both.
    """
    found = types(inst)
    return found[-1] if found else None


def types(inst: ir.Instruction) -> list[str]:
    return [m for m in inst.modifiers if _is_type(m)]


def _is_type(mod: str) -> bool:
    return mod in FLOAT_TYPES or mod == "pred" or (mod[:1] in "bsuf" and mod[1:].isdigit())


def memory_space(inst: ir.Instruction) -> str | None:
    """State space of a memory instruction, or None for non-memory ones.

    ``ld.shared::cta.u32`` -> ``shared``; ``ld.u32`` -> ``generic``.
    """
    if inst.opcode not in MEMORY_OPS:
        return None
    for mod in inst.modifiers:
        base = mod.split("::")[0]
        if base in STATE_SPACES:
            return base
    return "generic"


def memory_kind(inst: ir.Instruction) -> str | None:
    """``load``, ``store``, ``atomic`` or ``other`` for memory instructions."""
    if inst.opcode not in MEMORY_OPS:
        return None
    if inst.opcode in ("ld", "ldu"):
        return "load"
    if inst.opcode == "st":
        return "store"
    if inst.opcode in ("atom", "red"):
        return "atomic"
    return "other"


def is_branch(inst: ir.Instruction) -> bool:
    return inst.opcode in ("bra", "brx")


def ends_function(inst: ir.Instruction) -> bool:
    """Instructions after which control does not fall through."""
    return inst.opcode in ("ret", "exit", "trap") and inst.predicate is None
