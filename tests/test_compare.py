from pathlib import Path

from kernelsight.analysis import analyze
from kernelsight.analysis.resources import parse_ptxas_output
from kernelsight.compare import compare, normalized_listing
from kernelsight.ptx import parse

FIXTURES = Path(__file__).parent / "fixtures" / "ptx"


def load(name):
    return analyze(parse((FIXTURES / f"{name}.ptx").read_text()))


def kernel(body):
    return parse(f".entry k()\n{{\n{body}\n}}")


def test_normalization_ignores_register_and_label_numbering():
    a = kernel("add.s32 %r5, %r6, %r7;\n@%p3 bra $L__BB0_9;\n$L__BB0_9:\nret;")
    b = kernel("add.s32 %r1, %r2, %r3;\n@%p1 bra $L__BB0_2;\n$L__BB0_2:\nret;")
    assert normalized_listing(a.functions[0]) == normalized_listing(b.functions[0])
    assert compare(analyze(a), analyze(b)).functions[0].status == "identical"


def test_normalization_keeps_real_changes_and_special_registers():
    a = kernel("mov.u32 %r1, %tid.x;\nret;")
    b = kernel("mov.u32 %r1, %ctaid.x;\nret;")
    assert normalized_listing(a.functions[0]) == ["    mov.u32 %r1, %tid.x;", "    ret;"]
    assert compare(analyze(a), analyze(b)).functions[0].status == "changed"


def test_fast_math_changes():
    d = compare(load("math_heavy.sm89"), load("math_heavy.sm89_fastmath"))
    smooth = next(f for f in d.functions if f.name == "_Z6smoothf")
    assert smooth.mnemonics["div.rn.f32"] == (1, 0)
    assert smooth.mnemonics["div.approx.ftz.f32"] == (0, 1)
    assert smooth.findings_removed == ["precise-f32-math"]
    assert any(line.startswith("+") and "div.approx" in line for line in smooth.diff)

    heavy = next(f for f in d.functions if f.name == "math_heavy")
    before, after = heavy.metrics["instructions"]
    assert after < before / 2
    poly = next(f for f in d.functions if f.name == "polynomial")
    assert poly.status == "identical"  # f64 code is not affected by fast math


def test_ptxas_only_status():
    """Identical PTX assembled for two architectures gives different machine code."""
    a, b = load("matmul.sm75"), load("matmul.sm89")
    a.function("matmul_naive").resources = parse_ptxas_output(
        "Compiling entry function 'matmul_naive' for 'sm_75'\nUsed 49 registers"
    )["matmul_naive"]
    b.function("matmul_naive").resources = parse_ptxas_output(
        "Compiling entry function 'matmul_naive' for 'sm_89'\nUsed 40 registers"
    )["matmul_naive"]
    by_name = {f.name: f for f in compare(a, b).functions}
    assert by_name["matmul_naive"].status == "ptxas_only"
    assert by_name["matmul_naive"].metrics["ptxas_registers"] == (49, 40)
    assert by_name["matmul_tiled"].status == "identical"  # no ptxas data on either side


def test_target_change_with_identical_ptx():
    """For these kernels NVRTC emits the same PTX for sm_75 and sm_89; only the
    header differs. Architecture-specific work happens later, in ptxas."""
    d = compare(load("matmul.sm75"), load("matmul.sm89"))
    assert d.header == {"target": ("sm_75", "sm_89")}
    assert all(f.status == "identical" for f in d.functions)


def test_debug_build_adds_function_and_generic_loads():
    d = compare(load("elementwise.sm89"), load("elementwise.sm89_debug"))
    by_name = {f.name: f for f in d.functions}
    assert by_name["_Z11make_float4ffff"].status == "only_in_b"
    va = by_name["vector_add"]
    assert va.memory["load.global"] == (2, 0)
    assert va.memory["load.generic"] == (0, 2)
    # -G disables optimisations, including contracting a*x + y into an fma.
    assert by_name["saxpy"].mnemonics["fma.rn.f32"] == (1, 0)
