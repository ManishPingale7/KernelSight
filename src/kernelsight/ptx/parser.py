"""Parse PTX source into the IR defined in :mod:`kernelsight.ptx.ir`.

This is a hand-written recursive-descent parser: one method per grammar
rule, reading from the token list produced by the lexer. It covers the PTX
that NVCC/NVRTC generate (kernels, device functions, call sequences, debug
line info) and is written to be *tolerant*:

* unknown opcodes are kept as ordinary instructions (the opcode table lives
  in the analysis layer, not here);
* operand syntax the IR does not model is kept verbatim as ``RawOperand``;
* unknown directives are kept as ``Directive`` with their source text.

It raises :class:`PTXSyntaxError` (with a line number) only for input that
is structurally broken, such as a missing ``;`` or an unclosed ``{``.

Grammar sketch (simplified)::

    module      := header* item*
    header      := ".version" N | ".target" WORD ("," WORD)* | ".address_size" N
    item        := function | declaration ";" | directive
    function    := linkage* (".entry" | ".func") ["(" params ")"] NAME
                   ["(" params ")"] perf_directive* ( ";" | "{" body "}" )
    body        := ( label | declaration ";" | directive | "{" | "}"
                   | ["@" ["!"] REG] instruction )*
    instruction := OPCODE[.mod]* [operand ("," operand)*] ";"
"""

from __future__ import annotations

from kernelsight.ptx import ir
from kernelsight.ptx.lexer import PUNCT, STRING, WORD, PTXSyntaxError, Token, tokenize

# State spaces that start a variable declaration (PTX ISA 5.1).
STATE_SPACES = {".reg", ".sreg", ".const", ".global", ".local", ".param", ".shared", ".tex"}
LINKAGE = {".visible", ".extern", ".weak", ".common"}
# Directives that end at the end of the line rather than at a ';'.
LINE_DIRECTIVES = {".file", ".loc"}


def parse(source: str) -> ir.Module:
    """Parse PTX text into a :class:`~kernelsight.ptx.ir.Module`."""
    return _Parser(source).module()


class _Parser:
    def __init__(self, source: str):
        self.src = source
        self.toks: list[Token] = tokenize(source)
        self.pos = 0

    # ---------------------------------------------------------------- helpers

    def peek(self, ahead: int = 0) -> Token | None:
        i = self.pos + ahead
        return self.toks[i] if i < len(self.toks) else None

    def at(self, text: str, ahead: int = 0) -> bool:
        tok = self.peek(ahead)
        return tok is not None and tok.text == text and tok.kind != STRING

    def next(self) -> Token:
        tok = self.peek()
        if tok is None:
            last = self.toks[-1].line if self.toks else 1
            raise PTXSyntaxError("unexpected end of input", last)
        self.pos += 1
        return tok

    def expect(self, text: str) -> Token:
        tok = self.next()
        if tok.text != text:
            raise PTXSyntaxError(f"expected {text!r}, found {tok.text!r}", tok.line)
        return tok

    def text_between(self, first: Token, last: Token) -> str:
        """Original source text from the start of ``first`` to the end of ``last``."""
        return self.src[first.start : last.end]

    def skip_to(self, stop: str) -> Token:
        """Consume tokens up to and including ``stop`` at bracket depth 0."""
        depth = 0
        while True:
            tok = self.next()
            if tok.kind == PUNCT:
                if tok.text in "{[(":
                    depth += 1
                elif tok.text in "}])":
                    depth -= 1
                elif tok.text == stop and depth == 0:
                    return tok
            if depth < 0:
                raise PTXSyntaxError(f"unbalanced {tok.text!r} before {stop!r}", tok.line)

    def rest_of_line(self) -> Token:
        """Consume tokens until the end of the current line; return the last."""
        tok = self.next()
        while self.peek() is not None and self.peek().line == tok.line:
            tok = self.next()
        return tok

    # ----------------------------------------------------------------- module

    def module(self) -> ir.Module:
        mod = ir.Module()
        while (tok := self.peek()) is not None:
            text = tok.text
            if text == ".version":
                self.next()
                mod.version = self.next().text
            elif text == ".target":
                self.next()
                targets = [self.next().text]
                while self.at(","):
                    self.next()
                    targets.append(self.next().text)
                mod.target = tuple(targets)
            elif text == ".address_size":
                self.next()
                mod.address_size = int(self.next().text)
            elif text in LINE_DIRECTIVES:
                mod.items.append(self.line_directive())
            elif text == ".section":
                mod.items.append(self.section())
            elif text in LINKAGE or text in (".entry", ".func"):
                mod.items.append(self.function_or_variable())
            elif text in STATE_SPACES:
                mod.items.append(self.declaration())
            elif text.startswith("."):
                mod.items.append(self.directive())
            else:
                raise PTXSyntaxError(f"unexpected {text!r} at module level", tok.line)
        return mod

    def line_directive(self) -> ir.Directive:
        first = self.peek()
        last = self.rest_of_line()
        return ir.Directive(first.text, self.text_between(first, last), first.line)

    def directive(self) -> ir.Directive:
        """A directive terminated by ';' (e.g. ``.pragma "nounroll";``)."""
        first = self.peek()
        last = self.skip_to(";")
        return ir.Directive(first.text, self.text_between(first, last), first.line)

    def section(self) -> ir.Directive:
        """A DWARF debug section ``.section .debug_info { ... }`` (kept opaque)."""
        first = self.next()
        self.next()  # section name
        self.expect("{")
        depth = 1
        while depth:
            tok = self.next()
            if tok.kind == PUNCT and tok.text == "{":
                depth += 1
            elif tok.kind == PUNCT and tok.text == "}":
                depth -= 1
        return ir.Directive(".section", self.text_between(first, tok), first.line)

    # ------------------------------------------------------------ declarations

    def declaration(self, terminators: tuple[str, ...] = (";",)) -> ir.Declaration:
        """Parse ``[linkage] .space [.align N] [.type...] name[<N>][[dims]] [= init]``.

        Stops before (and does not consume) a terminator at bracket depth 0;
        for ';'-terminated declarations the ';' is consumed and included.
        """
        first = self.peek()
        depth = 0
        start = self.pos
        while True:
            tok = self.peek()
            if tok is None:
                raise PTXSyntaxError("unterminated declaration", first.line)
            if tok.kind == PUNCT:
                if depth == 0 and tok.text in terminators:
                    break
                if tok.text in "{[(":
                    depth += 1
                elif tok.text in "}])":
                    depth -= 1
            self.pos += 1
        toks = self.toks[start : self.pos]
        last = toks[-1]
        if self.at(";"):
            last = self.next()
        return _build_declaration(toks, self.text_between(first, last), first.line)

    def param_list(self) -> list[ir.Declaration]:
        """``( .param .u64 a, .param .u32 b )``"""
        self.expect("(")
        params = []
        while not self.at(")"):
            params.append(self.declaration(terminators=(",", ")")))
            if self.at(","):
                self.next()
        self.expect(")")
        return params

    # --------------------------------------------------------------- functions

    def function_or_variable(self) -> ir.Function | ir.Declaration:
        start = self.pos
        linkage = []
        while self.peek() is not None and self.peek().text in LINKAGE:
            linkage.append(self.next().text)
        if not (self.at(".entry") or self.at(".func")):
            # e.g. `.extern .shared .align 16 .b8 sdata[];` is a variable.
            self.pos = start
            return self.declaration()

        kind_tok = self.next()
        fn = ir.Function(
            name="", kind=kind_tok.text[1:], linkage=tuple(linkage), line=kind_tok.line
        )
        if fn.kind == "func" and self.at("("):
            fn.return_params = self.param_list()
        fn.name = self.next().text
        if self.at("("):
            fn.params = self.param_list()

        # Performance-tuning directives such as `.maxntid 256, 1, 1`.
        while not (self.at("{") or self.at(";")):
            first = self.next()
            last = first
            while self.peek() is not None and (
                self.at(",") or not self.peek().text.startswith(".")
            ):
                if self.at("{") or self.at(";"):
                    break
                last = self.next()
            fn.directives.append(
                ir.Directive(first.text, self.text_between(first, last), first.line)
            )

        if self.at(";"):
            self.next()  # prototype only, no body
        else:
            self.expect("{")
            fn.body = self.body()
        return fn

    def body(self) -> list[ir.Statement]:
        stmts: list[ir.Statement] = []
        depth = 0
        while True:
            tok = self.peek()
            if tok is None:
                raise PTXSyntaxError("missing '}' at end of function", self.toks[-1].line)
            text = tok.text
            if tok.kind == PUNCT and text == "}":
                self.next()
                if depth == 0:
                    return stmts
                depth -= 1
                stmts.append(ir.ScopeMarker(False, tok.line))
            elif tok.kind == PUNCT and text == "{":
                self.next()
                depth += 1
                stmts.append(ir.ScopeMarker(True, tok.line))
            elif tok.kind == WORD and self.at(":", 1) and not text.startswith("."):
                if self.at(".callprototype", 2):
                    # `prototype_0 : .callprototype (...) _ (...);`
                    last = self.skip_to(";")
                    stmts.append(
                        ir.Directive(".callprototype", self.text_between(tok, last), tok.line)
                    )
                else:
                    self.pos += 2
                    stmts.append(ir.Label(text, tok.line))
            elif text in STATE_SPACES:
                stmts.append(self.declaration())
            elif text in LINE_DIRECTIVES:
                stmts.append(self.line_directive())
            elif text.startswith("."):
                stmts.append(self.directive())
            else:
                stmts.append(self.instruction())

    # ------------------------------------------------------------ instructions

    def instruction(self) -> ir.Instruction:
        first = self.peek()
        predicate = None
        if self.at("@"):
            self.next()
            negated = self.at("!")
            if negated:
                self.next()
            predicate = ir.Predicate(ir.Register(self.next().text), negated)

        op_tok = self.next()
        if op_tok.kind != WORD:
            raise PTXSyntaxError(f"expected an instruction, found {op_tok.text!r}", op_tok.line)
        opcode, *modifiers = op_tok.text.split(".")
        operands = self.operand_list(";") if not self.at(";") else []
        self.expect(";")
        return ir.Instruction(opcode, tuple(modifiers), operands, predicate, first.line)

    def operand_list(self, closer: str) -> list[ir.Operand]:
        """Operands separated by ',' up to (not including) ``closer``."""
        operands = [self.operand(closer)]
        while self.at(","):
            self.next()
            operands.append(self.operand(closer))
        return operands

    def operand(self, closer: str) -> ir.Operand:
        start = self.pos
        try:
            op = self.simple_operand(closer)
        except (PTXSyntaxError, ValueError):
            op = None
        if op is not None and (self.at(",") or self.at(closer)):
            return op
        # Unmodelled syntax: keep the operand's source text verbatim.
        self.pos = start
        depth = 0
        first = last = self.peek()
        while True:
            tok = self.peek()
            if tok is None:
                raise PTXSyntaxError("unterminated operand list", first.line)
            if tok.kind == PUNCT:
                if depth == 0 and tok.text in (",", closer):
                    break
                if tok.text in "{[(":
                    depth += 1
                elif tok.text in "}])":
                    depth -= 1
            last = self.next()
        return ir.RawOperand(self.text_between(first, last))

    def simple_operand(self, closer: str) -> ir.Operand | None:
        tok = self.peek()
        if tok is None:
            return None
        if self.at("["):
            return self.address()
        if self.at("{"):
            self.next()
            elems = self.operand_list("}") if not self.at("}") else []
            self.expect("}")
            return ir.Vector(tuple(elems))
        if self.at("("):
            self.next()
            elems = self.operand_list(")") if not self.at(")") else []
            self.expect(")")
            return ir.ParenList(tuple(elems))
        if self.at("!"):
            self.next()
            return ir.Negated(self.simple_operand(closer))
        if self.at("-"):
            self.next()
            num = self.next()
            return ir.Immediate("-" + num.text) if _is_number(num.text) else None
        if tok.kind != WORD:
            return None
        self.next()
        op = _word_operand(tok.text)
        if self.at("|"):
            self.next()
            return ir.PredPair(op, _word_operand(self.next().text))
        return op

    def address(self) -> ir.Address | None:
        """``[%rd1]``, ``[%rd1+8]``, ``[param0+0]``, ``[sym]``, ``[64]``"""
        self.expect("[")
        base_tok = self.next()
        if _is_number(base_tok.text):
            base, offset = None, int(base_tok.text, 0)
        else:
            base, offset = _word_operand(base_tok.text), 0
            if self.at("+") or self.at("-"):
                sign = -1 if self.next().text == "-" else 1
                if self.at("-"):  # `[%rd1+-8]`
                    self.next()
                    sign = -sign
                offset = sign * int(self.next().text, 0)
        if not self.at("]"):
            return None
        self.next()
        if isinstance(base, ir.Immediate):
            return None
        return ir.Address(base, offset)


# --------------------------------------------------------------------------
# Small helpers (module level so they are easy to test on their own)
# --------------------------------------------------------------------------


def _is_number(text: str) -> bool:
    return text[:1].isdigit()


def _word_operand(text: str) -> ir.Operand:
    if text.startswith("%"):
        return ir.Register(text)
    if _is_number(text):
        return ir.Immediate(text)
    return ir.Symbol(text)


def _build_declaration(toks: list[Token], text: str, line: int) -> ir.Declaration:
    """Interpret the tokens of one declaration (see ``ir.Declaration``)."""
    state_space = None
    dtype = None
    align = None
    names: list[str] = []
    count = None
    array: list[int | None] = []
    i = 0
    while i < len(toks):
        t = toks[i].text
        if t == "=":
            break  # initializer: kept in `text`, not interpreted
        if t == ".align":
            align = int(toks[i + 1].text, 0)
            i += 2
            continue
        if t in STATE_SPACES and state_space is None:
            state_space = t[1:]
        elif t.startswith(".") and not names:
            if t not in LINKAGE and t not in (".ptr",) and t not in STATE_SPACES:
                dtype = t[1:]  # the last type-like directive wins (.v4 .f32 -> f32)
        elif t == "<" and names:
            count = int(toks[i + 1].text, 0)
            i += 3
            continue
        elif t == "[" and names:
            if toks[i + 1].text == "]":
                array.append(None)  # unsized: `sdata[]`
                i += 2
            else:
                array.append(int(toks[i + 1].text, 0))
                i += 3
            continue
        elif toks[i].kind == WORD and not t.startswith(".") and t not in (",", ";"):
            names.append(t)
        i += 1
    if state_space is None or not names:
        raise PTXSyntaxError(f"cannot parse declaration {text!r}", line)
    return ir.Declaration(
        state_space=state_space,
        type=dtype,
        name=names[0],
        text=text,
        count=count,
        array=tuple(array),
        align=align,
        line=line,
        others=tuple(names[1:]),
    )
