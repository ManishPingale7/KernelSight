"""Split PTX source text into tokens.

PTX is a line-oriented assembly language, but its grammar is token based:
whitespace and newlines only separate tokens. The lexer produces three kinds
of token:

* WORD    - anything built from letters, digits and ``_ $ % .``, including
            dotted names. One WORD can be an opcode with modifiers
            (``ld.global.f32``), a directive (``.reg``), a register
            (``%r1``, ``%tid.x``), a label (``$L__BB0_2``), a number
            (``42``, ``0x1F``, ``0f3F800000``) or a state space with a
            sub-qualifier (``.shared::cta``).
* STRING  - a double-quoted string (used by ``.file`` and ``.pragma``).
* PUNCT   - a single punctuation character such as ``; , { } [ ] ( ) < >``.

Comments (``// ...`` and ``/* ... */``) are skipped. Each token records its
line number and its character offsets, so the parser can recover the exact
original text of a declaration for faithful printing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD, STRING, PUNCT = "WORD", "STRING", "PUNCT"


class PTXSyntaxError(ValueError):
    def __init__(self, message: str, line: int):
        super().__init__(f"line {line}: {message}")
        self.line = line


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    line: int
    start: int  # offset of the first character in the source
    end: int  # offset one past the last character

    def __repr__(self) -> str:
        return f"{self.kind}({self.text!r}@{self.line})"


# Order matters: comments and whitespace first, then strings, then words.
_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<line_comment>//[^\n]*)
  | (?P<block_comment>/\*.*?\*/)
  | (?P<string>"(?:[^"\\\n]|\\.)*")
  | (?P<word>[A-Za-z0-9_$%.]+(?:::[A-Za-z0-9_$%.]+)*)
  | (?P<unterminated>/\*|")
  | (?P<punct>[^\sA-Za-z0-9_$%."])
    """,
    re.VERBOSE | re.DOTALL,
)


def tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    pos, line = 0, 1
    while pos < len(source):
        m = _TOKEN_RE.match(source, pos)
        kind, text = m.lastgroup, m.group()  # the regex matches any character
        if kind == "unterminated":
            what = "block comment" if text == "/*" else "string"
            raise PTXSyntaxError(f"unterminated {what}", line)
        if kind == "string":
            tokens.append(Token(STRING, text, line, pos, m.end()))
        elif kind == "word":
            tokens.append(Token(WORD, text, line, pos, m.end()))
        elif kind == "punct":
            tokens.append(Token(PUNCT, text, line, pos, m.end()))
        line += text.count("\n")
        pos = m.end()
    return tokens
