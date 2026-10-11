"""Intermediate representation (IR) for a PTX module.

The IR keeps exactly what the analyses need and no more:

    Module                       one .ptx file
      header (version, target, address_size)
      items (in source order):
        Declaration              module-scope variables (.global/.const/.shared)
        Function                 .entry kernels and .func device functions
        Directive                e.g. .file
        params: [Declaration]
        body: [Statement]        Instruction | Label | Declaration | Directive

Every node keeps the source line it came from, so analysis findings can point
back at the exact PTX line. Declarations and directives also keep their
original source text so the printer can reproduce them exactly.

Reference: NVIDIA "Parallel Thread Execution ISA", chapters 4 (Syntax),
5 (State Spaces, Types, and Variables) and 11 (Directives).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Operands
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Register:
    """A register such as ``%r1``, ``%fd3``, ``%p2`` or special ``%tid.x``."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Immediate:
    """A numeric literal: ``4``, ``-1``, ``0x10``, ``0f3F800000`` (float bits)."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Symbol:
    """A name: a label (``$L__BB0_2``), variable (``smem``) or function."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Address:
    """A memory address ``[base]`` or ``[base+offset]``.

    ``base`` is a register, a symbol, or None for an absolute address ``[64]``.
    """

    base: Register | Symbol | None
    offset: int = 0

    def __str__(self) -> str:
        if self.base is None:
            return f"[{self.offset}]"
        if self.offset == 0:
            return f"[{self.base}]"
        # PTX writes negative offsets as `[%rd1+-4]`; ptxas rejects `[%rd1-4]`.
        return f"[{self.base}+{self.offset}]"


@dataclass(frozen=True)
class Vector:
    """A brace list ``{%f1, %f2, %f3, %f4}`` used by vector loads/stores."""

    elements: tuple[Operand, ...]

    def __str__(self) -> str:
        return "{" + ", ".join(str(e) for e in self.elements) + "}"


@dataclass(frozen=True)
class ParenList:
    """A parenthesised list, used by ``call``: ``call (ret), f, (a, b);``."""

    elements: tuple[Operand, ...]

    def __str__(self) -> str:
        return "(" + ", ".join(str(e) for e in self.elements) + ")"


@dataclass(frozen=True)
class Negated:
    """A negated predicate operand such as ``!%p1``."""

    operand: Operand

    def __str__(self) -> str:
        return f"!{self.operand}"


@dataclass(frozen=True)
class PredPair:
    """``setp`` can write two predicates at once: ``%p1|%p2``."""

    first: Operand
    second: Operand

    def __str__(self) -> str:
        return f"{self.first}|{self.second}"


@dataclass(frozen=True)
class RawOperand:
    """Anything the parser does not model (kept verbatim, never dropped)."""

    text: str

    def __str__(self) -> str:
        return self.text


Operand = (
    Register | Immediate | Symbol | Address | Vector | ParenList | Negated | PredPair | RawOperand
)

# --------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Predicate:
    """Guard on an instruction: ``@%p1`` executes only where %p1 is true,
    ``@!%p1`` only where it is false."""

    register: Register
    negated: bool = False

    def __str__(self) -> str:
        return f"@{'!' if self.negated else ''}{self.register}"


@dataclass
class Instruction:
    """One PTX instruction, e.g. ``@%p1 ld.global.nc.f32 %f1, [%rd4+8];``.

    ``opcode`` is the base operation (``ld``) and ``modifiers`` the dotted
    suffixes (``("global", "nc", "f32")``).
    """

    opcode: str
    modifiers: tuple[str, ...]
    operands: list[Operand]
    predicate: Predicate | None = None
    line: int = 0

    @property
    def mnemonic(self) -> str:
        """Full dotted name, e.g. ``ld.global.nc.f32``."""
        return ".".join((self.opcode, *self.modifiers))

    def __str__(self) -> str:
        guard = f"{self.predicate} " if self.predicate else ""
        ops = ", ".join(str(o) for o in self.operands)
        return f"{guard}{self.mnemonic}{' ' + ops if ops else ''};"


@dataclass
class Label:
    """A branch target, e.g. ``$L__BB0_2:``."""

    name: str
    line: int = 0

    def __str__(self) -> str:
        return f"{self.name}:"


@dataclass
class Declaration:
    """A variable or register declaration.

    Examples::

        .reg .f32 %f<4>;                 state_space="reg", count=4
        .shared .align 4 .b8 tile[1024];  state_space="shared", array=[1024]
        .param .u64 vecAdd_param_0        (a kernel parameter, no ';')

    ``count`` is the N in a parameterised register list ``%f<N>`` (which
    declares %f0 .. %f{N-1}); ``array`` holds array dimensions. ``text`` is
    the original source, used by the printer.
    """

    state_space: str
    type: str | None
    name: str
    text: str
    count: int | None = None
    array: tuple[int | None, ...] = ()
    others: tuple[str, ...] = ()  # further names in ``.reg .b32 %a, %b;``
    align: int | None = None
    line: int = 0

    def __str__(self) -> str:
        return self.text


@dataclass
class Directive:
    """A directive the IR keeps but does not interpret, e.g. ``.pragma
    "nounroll";``, ``.loc 1 12 3``, or a function's ``.maxntid 256, 1, 1``."""

    name: str
    text: str
    line: int = 0

    def __str__(self) -> str:
        return self.text


@dataclass
class ScopeMarker:
    """``{`` or ``}`` opening/closing a nested block inside a function body.

    Nested blocks give registers declared inside them a limited scope; they
    are common around ``call`` sequences.
    """

    opening: bool
    line: int = 0

    def __str__(self) -> str:
        return "{" if self.opening else "}"


Statement = Instruction | Label | Declaration | Directive | ScopeMarker

# --------------------------------------------------------------------------
# Functions and modules
# --------------------------------------------------------------------------


@dataclass
class Function:
    """A kernel (``.entry``) or device function (``.func``).

    ``body`` is None for a prototype/declaration without a body (for example
    ``.extern .func foo(...);``).
    """

    name: str
    kind: str  # "entry" or "func"
    linkage: tuple[str, ...] = ()  # e.g. (".visible",) or (".extern",)
    params: list[Declaration] = field(default_factory=list)
    return_params: list[Declaration] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)  # .maxntid etc.
    body: list[Statement] | None = None
    line: int = 0

    @property
    def is_kernel(self) -> bool:
        return self.kind == "entry"

    @property
    def instructions(self) -> list[Instruction]:
        return [s for s in (self.body or []) if isinstance(s, Instruction)]


@dataclass
class Module:
    """A whole .ptx file.

    ``items`` keeps module-level declarations, functions and directives in
    source order: PTX requires a variable or function prototype to appear
    before it is used, so the printer must preserve that order.
    """

    version: str | None = None  # PTX ISA version, e.g. "8.5"
    target: tuple[str, ...] = ()  # e.g. ("sm_89",) or ("sm_80", "debug")
    address_size: int | None = None  # 32 or 64
    items: list[Declaration | Function | Directive] = field(default_factory=list)

    @property
    def functions(self) -> list[Function]:
        return [i for i in self.items if isinstance(i, Function)]

    @property
    def globals(self) -> list[Declaration]:
        return [i for i in self.items if isinstance(i, Declaration)]

    @property
    def kernels(self) -> list[Function]:
        return [f for f in self.functions if f.is_kernel and f.body is not None]

    def function(self, name: str) -> Function:
        for f in self.functions:
            if f.name == name:
                return f
        raise KeyError(name)
