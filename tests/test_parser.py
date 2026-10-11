import pytest

from kernelsight.ptx import PTXSyntaxError, ir, parse

KERNEL = """
.version 8.5
.target sm_89
.address_size 64

.extern .shared .align 16 .b8 sdata[];
.global .align 4 .b8 table[4] = {1, 2, 3, 4};

.visible .entry k(
    .param .u64 k_param_0,
    .param .u32 k_param_1
)
.maxntid 256, 1, 1
{
    .reg .pred %p<3>;
    .reg .f32 %f<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;
    .shared .align 4 .b8 tile[1024];

    ld.param.u64 %rd1, [k_param_0];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 32;
    @%p1 bra $L__BB0_2;
    @!%p2 bra.uni $L__BB0_2;
    ld.global.nc.v4.f32 {%f0, %f1, %f2, %f3}, [%rd1+-16];
    fma.rn.f32 %f2, %f1, 0f3F800000, %f1;
    st.global.f32 [%rd1], %f2;
$L__BB0_2:
    bar.sync 0;
    ret;
}
"""


@pytest.fixture
def module():
    return parse(KERNEL)


def test_header(module):
    assert module.version == "8.5"
    assert module.target == ("sm_89",)
    assert module.address_size == 64


def test_module_globals(module):
    sdata, table = module.globals
    assert (sdata.state_space, sdata.name, sdata.array, sdata.align) == (
        "shared",
        "sdata",
        (None,),
        16,
    )
    assert (table.name, table.array, table.type) == ("table", (4,), "b8")
    assert table.text.endswith("= {1, 2, 3, 4};")


def test_kernel_signature(module):
    (k,) = module.kernels
    assert k.name == "k" and k.is_kernel and k.linkage == (".visible",)
    assert [(p.name, p.type) for p in k.params] == [("k_param_0", "u64"), ("k_param_1", "u32")]
    assert [d.text for d in k.directives] == [".maxntid 256, 1, 1"]


def test_register_declarations(module):
    regs = [
        s
        for s in module.kernels[0].body
        if isinstance(s, ir.Declaration) and s.state_space == "reg"
    ]
    assert [(r.name, r.type, r.count) for r in regs] == [
        ("%p", "pred", 3),
        ("%f", "f32", 4),
        ("%r", "b32", 5),
        ("%rd", "b64", 5),
    ]


def test_instructions(module):
    insts = module.kernels[0].instructions
    assert [i.mnemonic for i in insts] == [
        "ld.param.u64",
        "mov.u32",
        "setp.ge.u32",
        "bra",
        "bra.uni",
        "ld.global.nc.v4.f32",
        "fma.rn.f32",
        "st.global.f32",
        "bar.sync",
        "ret",
    ]
    ld = insts[5]
    assert ld.opcode == "ld" and ld.modifiers == ("global", "nc", "v4", "f32")
    assert ld.operands[0] == ir.Vector(tuple(ir.Register(f"%f{i}") for i in range(4)))
    assert ld.operands[1] == ir.Address(ir.Register("%rd1"), -16)
    assert insts[6].operands[2] == ir.Immediate("0f3F800000")
    assert insts[1].operands[1] == ir.Register("%tid.x")
    assert insts[-1].operands == []


def test_predicates(module):
    insts = module.kernels[0].instructions
    assert insts[3].predicate == ir.Predicate(ir.Register("%p1"), negated=False)
    assert insts[4].predicate == ir.Predicate(ir.Register("%p2"), negated=True)
    assert insts[3].operands == [ir.Symbol("$L__BB0_2")]
    assert str(insts[4]) == "@!%p2 bra.uni $L__BB0_2;"


def test_labels_and_lines(module):
    body = module.kernels[0].body
    labels = [s for s in body if isinstance(s, ir.Label)]
    assert [lb.name for lb in labels] == ["$L__BB0_2"]
    ret = module.kernels[0].instructions[-1]
    assert KERNEL.splitlines()[ret.line - 1].strip() == "ret;"


# --- Regressions for bugs in the original regex-based prototype -----------


def test_predicated_branch_is_an_instruction():
    """The prototype's `@\\w+` pattern could not match `@%p1`, so predicated
    branches were never counted."""
    fn = parse(".entry f() { @%p1 bra $L1; $L1: ret; }").functions[0]
    assert [i.mnemonic for i in fn.instructions] == ["bra", "ret"]


def test_declarations_are_not_instructions():
    """The prototype counted any line ending in ';' (e.g. `.shared ...;`)."""
    fn = parse(".entry f() { .shared .align 4 .b8 s[16]; .local .b8 l[8]; ret; }").functions[0]
    assert len(fn.instructions) == 1


# --- Call sequences, device functions, prototypes ------------------------

CALLS = """
.version 8.5
.target sm_89
.address_size 64
.extern .func (.param .b32 func_retval0) vprintf(.param .b64 p0, .param .b64 p1);
.func (.param .b32 func_retval0) square(.param .b32 square_param_0)
{
    .reg .f32 %f<3>;
    ld.param.f32 %f1, [square_param_0];
    mul.f32 %f2, %f1, %f1;
    st.param.f32 [func_retval0+0], %f2;
    ret;
}
.visible .entry caller()
{
    .reg .f32 %f<3>;
    .reg .b32 %r<4>;
    .reg .pred %p<3>;
    { // callseq 0, 0
    .reg .b32 temp_param_reg;
    .param .b32 param0;
    st.param.f32 [param0+0], %f1;
    .param .b32 retval0;
    call.uni (retval0),
    square,
    (
    param0
    );
    ld.param.f32 %f2, [retval0+0];
    }
    shfl.sync.down.b32 %r1|%p1, %r2, 16, 31, -1;
    ret;
}
"""


def test_functions_and_prototypes():
    m = parse(CALLS)
    vprintf, square, caller = m.functions
    assert vprintf.body is None and vprintf.linkage == (".extern",)
    assert [p.name for p in vprintf.return_params] == ["func_retval0"]
    assert square.kind == "func" and not square.is_kernel
    assert [k.name for k in m.kernels] == ["caller"]


def test_call_sequence():
    caller = parse(CALLS).function("caller")
    call = next(i for i in caller.instructions if i.opcode == "call")
    assert call.modifiers == ("uni",)
    assert call.operands == [
        ir.ParenList((ir.Symbol("retval0"),)),
        ir.Symbol("square"),
        ir.ParenList((ir.Symbol("param0"),)),
    ]
    scopes = [s for s in caller.body if isinstance(s, ir.ScopeMarker)]
    assert [s.opening for s in scopes] == [True, False]


def test_predicate_pair_and_negative_immediate():
    shfl = next(i for i in parse(CALLS).function("caller").instructions if i.opcode == "shfl")
    assert shfl.operands[0] == ir.PredPair(ir.Register("%r1"), ir.Register("%p1"))
    assert shfl.operands[-1] == ir.Immediate("-1")


# --- Tolerance and errors -------------------------------------------------


def test_unknown_opcode_is_kept():
    fn = parse(".entry f() { frobnicate.b32 %r1, %r2; ret; }").functions[0]
    assert fn.instructions[0].opcode == "frobnicate"


def test_unmodelled_operand_kept_verbatim():
    fn = parse(".entry f() { mov.u64 %rd1, table+8; ret; }").functions[0]
    assert fn.instructions[0].operands[1] == ir.RawOperand("table+8")


def test_line_directives_and_pragmas():
    src = '.file 1 "a.cu"\n.entry f()\n{\n.loc 1 2 3\n.pragma "nounroll";\nret;\n}'
    m = parse(src)
    assert m.items[0].text == '.file 1 "a.cu"'
    body = m.functions[0].body
    assert [type(s).__name__ for s in body] == ["Directive", "Directive", "Instruction"]
    assert body[1].text == '.pragma "nounroll";'


@pytest.mark.parametrize(
    "src, line",
    [
        (".entry f() {\n ret\n}", 3),  # missing ';' -> sees '}' where it expects ';'
        (".entry f() {\n ret;\n", 2),  # missing closing brace
        (".entry f() {\n ret;\n}\n%r1", 4),  # junk at module level
    ],
)
def test_syntax_errors_report_line(src, line):
    with pytest.raises(PTXSyntaxError) as exc:
        parse(src)
    assert exc.value.line == line
