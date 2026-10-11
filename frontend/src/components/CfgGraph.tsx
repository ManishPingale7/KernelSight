"use client";

import type { Block } from "@/lib/api";

// An "arc diagram" of the control-flow graph: basic blocks stacked in source
// order, fall-through edges as short arrows, forward jumps as arcs on the
// left and loop back edges as arcs on the right. Deterministic and readable
// for the block counts typical of a single kernel.

const NODE_W = 230;
const NODE_H = 38;
const GAP = 22;
const LEFT = 150; // room for forward arcs
const RIGHT = 150; // room for back-edge arcs

export function CfgGraph({ blocks, headers }: { blocks: Block[]; headers: Set<string> }) {
  const index = new Map(blocks.map((b, i) => [b.name, i]));
  const y = (i: number) => 10 + i * (NODE_H + GAP);
  const width = LEFT + NODE_W + RIGHT;
  const height = y(blocks.length) + 10;
  const x0 = LEFT;
  const x1 = LEFT + NODE_W;

  const edges: { from: number; to: number }[] = [];
  blocks.forEach((b, i) =>
    b.successors.forEach((s) => {
      const j = index.get(s);
      if (j !== undefined) edges.push({ from: i, to: j });
    }),
  );

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      style={{ maxWidth: width }}
      role="img"
      aria-label="Control-flow graph"
    >
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
        </marker>
        <marker id="arrow-back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
        </marker>
      </defs>

      {edges.map(({ from, to }) => {
        const key = `${from}-${to}`;
        if (to === from + 1) {
          // fall-through: straight down between consecutive blocks
          const x = x0 + NODE_W / 2;
          return (
            <line key={key} x1={x} y1={y(from) + NODE_H} x2={x} y2={y(to) - 2}
              stroke="var(--muted)" strokeWidth={1.4} markerEnd="url(#arrow)" />
          );
        }
        if (to === from) {
          // self-loop: a small loop on the right edge of the block
          const top = y(from) + NODE_H * 0.3;
          const bottom = y(from) + NODE_H * 0.7;
          return (
            <path key={key} d={`M${x1},${bottom} C${x1 + 34},${bottom + 10} ${x1 + 34},${top - 10} ${x1 + 2},${top}`}
              fill="none" stroke="var(--accent)" strokeWidth={1.8} markerEnd="url(#arrow-back)" />
          );
        }
        const back = to < from;
        const span = Math.abs(to - from);
        const reach = 18 + Math.min(span, 10) * 11;
        const ya = y(from) + NODE_H / 2;
        const yb = y(to) + NODE_H / 2;
        const xs = back ? x1 : x0;
        const xc = back ? x1 + reach : x0 - reach;
        return (
          <path key={key} d={`M${xs},${ya} C${xc},${ya} ${xc},${yb} ${xs + (back ? 2 : -2)},${yb}`}
            fill="none" stroke={back ? "var(--accent)" : "var(--muted)"}
            strokeWidth={back ? 1.8 : 1.3} markerEnd={back ? "url(#arrow-back)" : "url(#arrow)"} />
        );
      })}

      {blocks.map((b, i) => {
        const depthTint = Math.min(b.loop_depth, 3) * 0.12;
        return (
          <g key={b.id}>
            <title>
              {`${b.name}: ${b.instructions} instructions` +
                (b.lines ? `, PTX lines ${b.lines[0]}–${b.lines[1]}` : "") +
                `, loop depth ${b.loop_depth}`}
            </title>
            <rect x={x0} y={y(i)} width={NODE_W} height={NODE_H} rx={7}
              fill="var(--panel)" stroke={headers.has(b.name) ? "var(--accent)" : "var(--line)"}
              strokeWidth={headers.has(b.name) ? 2 : 1} />
            {depthTint > 0 && (
              <rect x={x0} y={y(i)} width={NODE_W} height={NODE_H} rx={7}
                fill="var(--accent)" opacity={depthTint} />
            )}
            <text x={x0 + 12} y={y(i) + 16} fontSize={12} fontFamily="var(--mono)" fill="var(--fg)">
              {b.name}
            </text>
            <text x={x0 + 12} y={y(i) + 30} fontSize={11} fill="var(--muted)">
              {`${b.instructions} instr` + (b.loop_depth ? ` · loop depth ${b.loop_depth}` : "")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
