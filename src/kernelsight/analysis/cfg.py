"""Control-flow graph (CFG), dominators and natural loops for one function.

Basic blocks
    A basic block is a straight-line run of instructions with one entry (the
    first instruction) and one exit (the last). A new block starts at every
    label (a possible branch target) and after every branch/return.

Edges
    ``bra L``           -> the block at L only
    ``@%p bra L``       -> the block at L, and the next block (fall-through)
    ``ret`` / ``exit``  -> no successor
    anything else       -> the next block

Dominators
    Block D dominates block B if every path from the entry to B passes
    through D. Computed with the classic iterative data-flow algorithm:
    dom(entry) = {entry};  dom(B) = {B} ∪ ⋂ dom(P) over predecessors P.

Natural loops
    An edge T -> H is a *back edge* if H dominates T. The natural loop of
    that edge is H plus every block that can reach T without going through
    H. Loops that share a header are merged. A block's *loop depth* is the
    number of loops containing it; depth > 0 means the instruction probably
    runs many times, which static counts alone do not show.

Limitations: only natural (reducible) loops are found, which covers code
generated from structured CUDA C++. Indirect branches (``brx.idx``) are
flagged but their targets are not resolved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelsight.analysis import opcodes
from kernelsight.ptx import ir


@dataclass
class BasicBlock:
    id: int
    labels: list[str] = field(default_factory=list)
    instructions: list[ir.Instruction] = field(default_factory=list)
    successors: list[int] = field(default_factory=list)
    predecessors: list[int] = field(default_factory=list)
    loop_depth: int = 0

    @property
    def name(self) -> str:
        return self.labels[0] if self.labels else f"B{self.id}"

    @property
    def lines(self) -> tuple[int, int] | None:
        if not self.instructions:
            return None
        return self.instructions[0].line, self.instructions[-1].line


@dataclass
class Loop:
    header: int
    blocks: frozenset[int]
    back_edges: list[tuple[int, int]]
    depth: int = 1  # 1 = outermost


@dataclass
class CFG:
    blocks: list[BasicBlock]
    dominators: dict[int, frozenset[int]]
    loops: list[Loop]
    unreachable: list[int]
    has_indirect_branch: bool = False
    unresolved_targets: list[str] = field(default_factory=list)

    @property
    def edges(self) -> list[tuple[int, int]]:
        return [(b.id, s) for b in self.blocks for s in b.successors]

    def block_of_line(self, line: int) -> BasicBlock | None:
        for b in self.blocks:
            if any(i.line == line for i in b.instructions):
                return b
        return None


def build_cfg(fn: ir.Function) -> CFG:
    blocks = _split_blocks(fn.body or [])
    by_label = {label: b.id for b in blocks for label in b.labels}
    cfg = CFG(blocks, {}, [], [])
    _connect(cfg, by_label)
    reachable = _reachable(blocks)
    cfg.unreachable = [b.id for b in blocks if b.id not in reachable]
    cfg.dominators = _dominators(blocks, reachable)
    cfg.loops = _natural_loops(blocks, cfg.dominators)
    for b in blocks:
        b.loop_depth = sum(1 for lp in cfg.loops if b.id in lp.blocks)
    return cfg


# --------------------------------------------------------------------------


def _split_blocks(body: list[ir.Statement]) -> list[BasicBlock]:
    blocks = [BasicBlock(0)]
    for stmt in body:
        current = blocks[-1]
        if isinstance(stmt, ir.Label):
            if current.instructions:  # a label starts a new block
                blocks.append(BasicBlock(len(blocks)))
            blocks[-1].labels.append(stmt.name)
        elif isinstance(stmt, ir.Instruction):
            current.instructions.append(stmt)
            if opcodes.is_branch(stmt) or opcodes.ends_function(stmt):
                blocks.append(BasicBlock(len(blocks)))
    # Drop a trailing empty block created after the final ret, unless it is
    # a labelled (branch-target) block.
    if len(blocks) > 1 and not blocks[-1].instructions and not blocks[-1].labels:
        blocks.pop()
    return blocks


def _connect(cfg: CFG, by_label: dict[str, int]) -> None:
    blocks = cfg.blocks

    def add_edge(src: int, dst: int) -> None:
        if dst not in blocks[src].successors:
            blocks[src].successors.append(dst)
            blocks[dst].predecessors.append(src)

    for b in blocks:
        nxt = b.id + 1 if b.id + 1 < len(blocks) else None
        last = b.instructions[-1] if b.instructions else None
        falls_through = True
        if last is not None and opcodes.is_branch(last):
            if last.opcode == "brx":
                cfg.has_indirect_branch = True
            else:
                target = str(last.operands[0]) if last.operands else ""
                if target in by_label:
                    add_edge(b.id, by_label[target])
                else:
                    cfg.unresolved_targets.append(target)
            falls_through = last.predicate is not None
        elif last is not None and opcodes.ends_function(last):
            falls_through = False
        if falls_through and nxt is not None:
            add_edge(b.id, nxt)


def _reachable(blocks: list[BasicBlock]) -> set[int]:
    seen, stack = set(), [0]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(blocks[n].successors)
    return seen


def _dominators(blocks: list[BasicBlock], reachable: set[int]) -> dict[int, frozenset[int]]:
    nodes = [b.id for b in blocks if b.id in reachable]
    dom = {n: set(nodes) for n in nodes}
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for n in nodes[1:]:
            preds = [p for p in blocks[n].predecessors if p in reachable]
            new = set.intersection(*(dom[p] for p in preds)) if preds else set()
            new = new | {n}
            if new != dom[n]:
                dom[n], changed = new, True
    return {n: frozenset(d) for n, d in dom.items()}


def _natural_loops(blocks: list[BasicBlock], dom: dict[int, frozenset[int]]) -> list[Loop]:
    by_header: dict[int, Loop] = {}
    for b in blocks:
        for s in b.successors:
            if b.id in dom and s in dom[b.id]:  # back edge b -> s
                body = _loop_body(blocks, header=s, tail=b.id)
                if s in by_header:
                    old = by_header[s]
                    by_header[s] = Loop(s, old.blocks | body, [*old.back_edges, (b.id, s)])
                else:
                    by_header[s] = Loop(s, body, [(b.id, s)])
    loops = sorted(by_header.values(), key=lambda lp: lp.header)
    for lp in loops:
        lp.depth = sum(1 for other in loops if lp.blocks < other.blocks) + 1
    return loops


def _loop_body(blocks: list[BasicBlock], header: int, tail: int) -> frozenset[int]:
    # Walk backwards from the tail; never walk past the header. For a
    # self-loop (tail == header) the body is just the header.
    body = {header, tail}
    stack = [tail] if tail != header else []
    while stack:
        n = stack.pop()
        for p in blocks[n].predecessors:
            if p not in body:
                body.add(p)
                stack.append(p)
    return frozenset(body)
