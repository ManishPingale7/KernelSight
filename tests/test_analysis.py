from pathlib import Path

import pytest

from kernelsight import toolchain
from kernelsight.analysis import analyze, opcodes, target_arch
from kernelsight.analysis.cfg import build_cfg
from kernelsight.analysis.mix import instruction_mix
from kernelsight.analysis.registers import analyze_registers, defs_uses, slots
from kernelsight.analysis.resources import parse_ptxas_output
from kernelsight.ptx import parse

FIXTURES = Path(__file__).parent / "fixtures" / "ptx"


def fn(body: str):
    """Parse a function body written in shorthand PTX."""
    return parse(f".entry f()\n{{\n{body}\n}}").functions[0]


def inst(text: str):
    return fn(text).instructions[0]


# --- opcode classification ------------------------------------------------


@pytest.mark.parametrize(
    "text, cat, space",
    [
        ("ld.global.nc.f32 %f1, [%rd1];", "memory", "global"),
        ("st.shared::cta.u32 [%r1], %r2;", "memory", "shared"),
        ("ld.u32 %r1, [%rd1];", "memory", "generic"),
        ("atom.global.add.f32 %f1, [%rd1], %f2;", "memory", "global"),
        ("fma.rn.f32 %f1, %f2, %f3, %f4;", "fp_arith", None),
        ("mad.lo.s32 %r1, %r2, %r3, %r4;", "int_arith", None),
        ("ex2.approx.ftz.f32 %f1, %f2;", "special_fn", None),
        ("setp.ge.s32 %p1, %r1, %r2;", "compare", None),
        ("cvta.to.global.u64 %rd1, %rd2;", "move_convert", None),
        ("bar.sync 0;", "sync", None),
        ("shfl.sync.down.b32 %r1|%p1, %r2, 1, 31, -1;", "warp", None),
        ("frobnicate.b32 %r1;", "other", None),
    ],
)
def test_classification(text, cat, space):
    i = inst(text)
    assert opcodes.category(i) == cat
    assert opcodes.memory_space(i) == space


def test_data_type():
    assert opcodes.data_type(inst("ld.global.v4.f32 {%f1,%f2,%f3,%f4}, [%rd1];")) == "f32"
    assert opcodes.types(inst("cvt.rn.f32.s32 %f1, %r1;")) == ["f32", "s32"]


def test_instruction_mix():
    mix = instruction_mix(
        fn("""
        ld.global.f32 %f1, [%rd1];
        ld.shared.f32 %f2, [%r1];
        add.f32 %f3, %f1, %f2;
        @%p1 st.global.f32 [%rd1], %f3;
        frob.b32 %r1;
        ret;
        """).instructions
    )
    assert mix.total == 6
    assert mix.memory_ops(kind="load") == 2
    assert mix.memory_ops(space="global") == 2
    assert mix.by_category["fp_arith"] == 1
    assert mix.predicated == 1
    assert mix.uncategorized == ["frob"]


# --- control-flow graph -----------------------------------------------------

DIAMOND = """
    setp.lt.s32 %p1, %r1, 0;
    @%p1 bra ELSE;
    add.s32 %r2, %r1, 1;
    bra.uni JOIN;
ELSE:
    sub.s32 %r2, %r1, 1;
JOIN:
    ret;
"""


def test_diamond_cfg_and_dominators():
    cfg = build_cfg(fn(DIAMOND))
    names = [b.name for b in cfg.blocks]
    assert names == ["B0", "B1", "ELSE", "JOIN"]
    assert sorted(cfg.edges) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    # Neither branch dominates the join; only the entry does.
    assert cfg.dominators[3] == {0, 3}
    assert cfg.loops == []


NESTED = """
    mov.u32 %r1, 0;
OUTER:
    mov.u32 %r2, 0;
INNER:
    add.s32 %r2, %r2, 1;
    setp.lt.s32 %p1, %r2, 8;
    @%p1 bra INNER;
    add.s32 %r1, %r1, 1;
    setp.lt.s32 %p2, %r1, 8;
    @%p2 bra OUTER;
    ret;
"""


def test_nested_loops():
    cfg = build_cfg(fn(NESTED))
    by_name = {b.name: b for b in cfg.blocks}
    assert [b.loop_depth for b in cfg.blocks] == [0, 1, 2, 1, 0]
    inner, outer = sorted(cfg.loops, key=lambda lp: len(lp.blocks))
    assert inner.header == by_name["INNER"].id and inner.depth == 2
    assert outer.header == by_name["OUTER"].id and outer.depth == 1


def test_self_loop_does_not_include_entry():
    """Regression: a block that branches to itself once pulled every block
    above it into the loop body."""
    cfg = build_cfg(fn("mov.u32 %r1, 0;\nL:\nadd.s32 %r1, %r1, 1;\n@%p1 bra L;\nret;"))
    (loop,) = cfg.loops
    assert loop.blocks == {loop.header}
    assert cfg.blocks[0].loop_depth == 0


def test_unreachable_code_and_unknown_target():
    cfg = build_cfg(fn("bra.uni END;\nadd.s32 %r1, %r1, 1;\nEND:\nret;\n@%p1 bra NOWHERE;"))
    assert cfg.unreachable == [1, 3]
    assert cfg.unresolved_targets == ["NOWHERE"]


def test_fixture_loops_match_source():
    """histogram256 has three sequential loops in the CUDA source."""
    module = parse((FIXTURES / "histogram.sm89.ptx").read_text())
    cfg = build_cfg(module.function("histogram256"))
    assert len(cfg.loops) == 3
    assert all(lp.depth == 1 for lp in cfg.loops)


# --- registers --------------------------------------------------------------


def test_slots():
    assert (slots("f32"), slots("b64"), slots("f64"), slots("pred")) == (1, 2, 2, 0)


def test_defs_and_uses():
    known = {n: "b32" for n in ("%r1", "%r2", "%r3")} | {"%p1": "pred", "%rd1": "b64"}
    assert defs_uses(inst("add.s32 %r1, %r2, %r3;"), known) == ({"%r1"}, {"%r2", "%r3"})
    # Stores write memory, not registers: every register operand is a use.
    assert defs_uses(inst("st.global.u32 [%rd1], %r1;"), known) == (set(), {"%rd1", "%r1"})
    # Special registers (%tid.x) are not virtual registers.
    assert defs_uses(inst("mov.u32 %r1, %tid.x;"), known) == ({"%r1"}, set())
    assert defs_uses(inst("@%p1 bra L;"), known) == (set(), {"%p1"})


def test_liveness_straight_line():
    f = fn("""
    .reg .b32 %r<5>;
    .reg .b64 %rd<2>;
    mov.u32 %r1, 1;
    mov.u32 %r2, 2;
    mov.u32 %r3, 3;
    add.s32 %r4, %r1, %r2;
    add.s32 %r4, %r4, %r3;
    mov.u64 %rd1, 0;
    st.global.u32 [%rd1], %r4;
    ret;
    """)
    info = analyze_registers(f, build_cfg(f))
    assert info.declared_count == 7  # %r0..%r4, %rd0, %rd1
    assert info.unused == {"%r0", "%rd0"}
    # After `mov %r3`: %r1, %r2, %r3 are live -> 3 slots. At the first add,
    # %r1 and %r2 die, so the result %r4 can reuse one of their registers
    # (operands that die are not counted alongside the result).
    # At st: %rd1 (2 slots) + %r4 (1) = 3.
    assert info.max_live_slots == 3


def test_liveness_counts_values_live_around_a_loop():
    f = fn("""
    .reg .b32 %r<3>;
    .reg .pred %p<2>;
    mov.u32 %r1, 0;
    mov.u32 %r2, 10;
L:
    add.s32 %r1, %r1, 1;
    setp.lt.s32 %p1, %r1, %r2;
    @%p1 bra L;
    ret;
    """)
    info = analyze_registers(f, build_cfg(f))
    # %r2 stays live across the whole loop together with %r1.
    assert info.max_live_slots == 2
    assert info.max_live_predicates == 1


# --- ptxas resource report --------------------------------------------------

# Real `ptxas -v` output for sm_89 (one kernel built with -maxrregcount=24), combined
# from several runs so a single sample covers spills, shared memory and a .func.
PTXAS_OUTPUT = (FIXTURES.parent / "ptxas" / "resource_report.txt").read_text()


def test_parse_ptxas_output():
    usage = parse_ptxas_output(PTXAS_OUTPUT)
    spill = usage["many_live_values"]
    assert (spill.registers, spill.stack_frame, spill.spill_stores, spill.spill_loads) == (
        24,
        352,
        416,
        576,
    )
    assert spill.spills and spill.arch == "sm_89"
    tiled = usage["matmul_tiled"]
    assert (tiled.registers, tiled.barriers, tiled.shared_mem) == (38, 1, 2048)
    assert tiled.const_mem == {0: 380, 2: 8}
    assert not tiled.spills
    assert usage["_Z6smoothf"].registers is None  # compiled with its caller


# --- findings ---------------------------------------------------------------


def _analyze_fixture(name, ptxas_output=None):
    return analyze(parse((FIXTURES / name).read_text()), ptxas_output)


def rules(fa):
    return {f.rule for f in fa.findings}


def test_findings_without_ptxas():
    heavy = _analyze_fixture("math_heavy.sm89.ptx")
    assert {"precise-f32-math", "fp64-arithmetic", "local-memory"} <= rules(
        heavy.function("math_heavy")
    )
    assert "fp64-arithmetic" in rules(heavy.function("polynomial"))
    hist = _analyze_fixture("histogram.sm89.ptx").function("histogram256")
    assert "atomics-in-loop" in rules(hist)
    # Every finding cites PTX lines that hold an instruction.
    lines = {i.line for i in hist.function.instructions}
    assert all(set(f.lines) <= lines for f in hist.findings)


def test_fast_math_removes_precise_math_finding():
    fast = _analyze_fixture("math_heavy.sm89_fastmath.ptx")
    assert all("precise-f32-math" not in rules(fa) for fa in fast.functions)


def test_clean_kernel_has_no_findings():
    assert _analyze_fixture("elementwise.sm89.ptx").function("vector_add").findings == []


@pytest.mark.toolchain
def test_spills_reported_with_low_register_limit():
    text = (FIXTURES / "register_pressure.sm89.ptx").read_text()
    module = parse(text)
    out = toolchain.run_ptxas(text, target_arch(module), ["-maxrregcount=24"])
    fa = analyze(module, out).function("many_live_values")
    assert fa.resources.registers <= 24
    assert "register-spills" in rules(fa)


@pytest.mark.toolchain
def test_ptxas_qualifies_local_memory_finding():
    text = (FIXTURES / "register_pressure.sm89.ptx").read_text()
    module = parse(text)
    fa = analyze(module, toolchain.run_ptxas(text, "sm_89")).function("dynamic_local_index")
    (finding,) = [f for f in fa.findings if f.rule == "local-memory"]
    assert "0-byte stack frame" in finding.caveat
