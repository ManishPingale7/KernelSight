"""Round-trip tests over the compiler-generated PTX corpus.

For every fixture:

1. parse it and print it again;
2. the printed text must be a fixpoint (printing a re-parse changes nothing)
   and must contain the same instructions;
3. with the toolchain installed, ptxas must accept the printed PTX and
   report the same per-function resource usage as for the original.

Step 3 uses NVIDIA's own assembler as an independent oracle for the parser.
"""

import re
from pathlib import Path

import pytest

from kernelsight import toolchain
from kernelsight.ptx import parse, print_module

FIXTURES = sorted((Path(__file__).parent / "fixtures" / "ptx").glob("*.ptx"))


def _arch(path: Path) -> str:
    return "sm_75" if ".sm75" in path.name else "sm_89"


def _instructions(module):
    return [(f.name, str(i)) for f in module.functions for i in f.instructions]


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_print_is_a_fixpoint(path):
    module = parse(path.read_text())
    printed = print_module(module)
    reparsed = parse(printed)
    assert print_module(reparsed) == printed
    assert _instructions(reparsed) == _instructions(module)
    assert len(_instructions(module)) > 0


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_no_operand_falls_back_to_raw_text(path):
    """Compiler output should be fully modelled; RawOperand is for odd input."""
    module = parse(path.read_text())
    raw = [
        str(i)
        for f in module.functions
        for i in f.instructions
        if any(type(o).__name__ == "RawOperand" for o in i.operands)
    ]
    assert raw == []


def _resource_lines(ptxas_output: str) -> list[str]:
    """The 'Used N registers' / stack / spill lines, which identify the result."""
    keep = re.compile(r"Used \d+ registers|bytes stack frame|Compiling entry")
    return [ln.strip() for ln in ptxas_output.splitlines() if keep.search(ln)]


@pytest.mark.toolchain
@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.name)
def test_ptxas_accepts_printed_ptx(path):
    original = toolchain.run_ptxas(path.read_text(), _arch(path))
    printed = toolchain.run_ptxas(print_module(parse(path.read_text())), _arch(path))
    assert _resource_lines(printed) == _resource_lines(original)
