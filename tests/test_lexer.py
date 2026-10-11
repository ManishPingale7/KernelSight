import pytest

from kernelsight.ptx.lexer import PUNCT, STRING, WORD, PTXSyntaxError, tokenize


def texts(src):
    return [t.text for t in tokenize(src)]


def test_dotted_words_stay_together():
    assert texts("ld.global.nc.f32 %f1, [%rd2+8];") == [
        "ld.global.nc.f32",
        "%f1",
        ",",
        "[",
        "%rd2",
        "+",
        "8",
        "]",
        ";",
    ]


def test_special_registers_and_qualified_state_spaces():
    assert texts("mov.u32 %r1, %tid.x;") == ["mov.u32", "%r1", ",", "%tid.x", ";"]
    assert texts("ld.shared::cta.u32") == ["ld.shared::cta.u32"]


def test_label_colon_is_separate():
    assert texts("$L__BB0_2:") == ["$L__BB0_2", ":"]


def test_comments_are_skipped_and_lines_counted():
    toks = tokenize("// header\n/* multi\nline */ ret; // trailing\nexit;")
    assert [(t.text, t.line) for t in toks] == [("ret", 3), (";", 3), ("exit", 4), (";", 4)]


def test_token_kinds_and_offsets():
    src = '.file 1 "a.cu"'
    toks = tokenize(src)
    assert [t.kind for t in toks] == [WORD, WORD, STRING]
    assert src[toks[2].start : toks[2].end] == '"a.cu"'
    assert tokenize("{")[0].kind == PUNCT


@pytest.mark.parametrize("src", ["/* never closed", '.file 1 "never closed'])
def test_unterminated_input_is_an_error(src):
    with pytest.raises(PTXSyntaxError):
        tokenize(src)
