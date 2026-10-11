"""Print a :class:`~kernelsight.ptx.ir.Module` back to PTX text.

Instructions are printed from their parsed fields (opcode, modifiers,
predicate, operands), not from the original text. That makes the printer a
check on the parser: if parsing loses or garbles anything, the printed PTX
will differ or ``ptxas`` will reject it. The tests use this "round trip"
(parse -> print -> ptxas) as a verification oracle.

Declarations and opaque directives are printed from their original text.
Comments and blank lines are not preserved.
"""

from __future__ import annotations

from kernelsight.ptx import ir


def print_module(module: ir.Module) -> str:
    out: list[str] = []
    if module.version:
        out.append(f".version {module.version}")
    if module.target:
        out.append(f".target {', '.join(module.target)}")
    if module.address_size is not None:
        out.append(f".address_size {module.address_size}")
    for item in module.items:
        out.append("")
        if isinstance(item, ir.Function):
            out.extend(_function_lines(item))
        else:
            out.append(str(item))
    return "\n".join(out) + "\n"


def print_function(fn: ir.Function) -> str:
    return "\n".join(_function_lines(fn)) + "\n"


def _param_block(params: list[ir.Declaration]) -> str:
    if not params:
        return "()"
    return "(\n" + ",\n".join(f"\t{p}" for p in params) + "\n)"


def _function_lines(fn: ir.Function) -> list[str]:
    head = " ".join([*fn.linkage, f".{fn.kind}"])
    if fn.return_params:
        head += " (" + ", ".join(str(p) for p in fn.return_params) + ")"
    head += f" {fn.name}"
    if fn.params or fn.body is not None or fn.kind == "entry":
        head += _param_block(fn.params)
    lines = [head]
    lines.extend(str(d) for d in fn.directives)
    if fn.body is None:
        lines[-1] += ";"
        return lines

    lines.append("{")
    indent = 1
    for stmt in fn.body:
        if isinstance(stmt, ir.ScopeMarker) and not stmt.opening:
            indent -= 1
        prefix = "" if isinstance(stmt, ir.Label) else "\t" * indent
        lines.append(prefix + str(stmt))
        if isinstance(stmt, ir.ScopeMarker) and stmt.opening:
            indent += 1
    lines.append("}")
    return lines
