"use client";

import { useState } from "react";

import type { AnalysisReport, FunctionReport } from "@/lib/api";
import { CATEGORY_LABELS, ProvenanceList, Tile } from "@/components/common";
import { CfgGraph } from "@/components/CfgGraph";

export function AnalysisView({ report }: { report: AnalysisReport }) {
  const [selected, setSelected] = useState(0);
  const fn = report.functions[Math.min(selected, report.functions.length - 1)];
  const mod = report.module;

  return (
    <div className="flex flex-col gap-4">
      <div className="panel text-sm">
        <ProvenanceList provenance={report.provenance} />
        <p className="muted mt-1">
          PTX ISA {mod.version}, target <code>{mod.target.join(", ")}</code>, {mod.address_size}-bit
        </p>
      </div>

      {report.functions.length === 0 ? (
        <p className="muted">No functions with a body in this input.</p>
      ) : (
        <>
          <div className="flex flex-wrap gap-2" role="tablist" aria-label="Functions">
            {report.functions.map((f, i) => (
              <button
                key={f.name}
                role="tab"
                aria-selected={f === fn}
                className={`chip ${f === fn ? "active" : ""}`}
                onClick={() => setSelected(i)}
              >
                <code>{f.name}</code>
                {f.findings.length > 0 && <span className="badge">{f.findings.length}</span>}
              </button>
            ))}
          </div>
          <FunctionPanel key={fn.name} fn={fn} />
        </>
      )}
    </div>
  );
}

function FunctionPanel({ fn }: { fn: FunctionReport }) {
  const { instructions: ins, loops, registers: regs, ptxas } = fn;
  const max = Math.max(1, ...Object.values(ins.by_category));
  const headers = new Set(fn.cfg.loops.map((l) => l.header));

  return (
    <div className="flex flex-col gap-4">
      <div className="tiles">
        <Tile label="Instructions (static)" value={ins.total} />
        <Tile label="Inside loops" value={loops.instructions_in_loops} />
        <Tile label="Blocks / loops" value={`${fn.cfg.blocks.length} / ${loops.count}`} />
        <Tile label="Virtual regs declared" value={regs.declared} />
        <Tile
          label="Peak live (32-bit slots)"
          value={regs.peak_live_slots}
          hint={regs.peak_live_line ? `at PTX line ${regs.peak_live_line}` : undefined}
        />
        {ptxas ? (
          <>
            <Tile label="ptxas registers" value={ptxas.registers ?? "n/a"} hint="physical, per thread" />
            <Tile label="Spills (st / ld bytes)" value={`${ptxas.spill_stores} / ${ptxas.spill_loads}`}
              warn={ptxas.spill_stores + ptxas.spill_loads > 0} />
            <Tile label="Shared memory" value={`${ptxas.shared_mem} B`} />
          </>
        ) : (
          <Tile label="ptxas" value="not run" hint="enable for real register counts" />
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="panel">
          <h3 className="section-title">Instruction mix</h3>
          <table className="data">
            <thead>
              <tr><th>Category</th><th className="num">Count</th><th className="num">In loops</th><th className="w-2/5" /></tr>
            </thead>
            <tbody>
              {Object.entries(ins.by_category).map(([cat, n]) => (
                <tr key={cat}>
                  <td>{CATEGORY_LABELS[cat] ?? cat}</td>
                  <td className="num">{n}</td>
                  <td className="num">{loops.by_category_in_loops[cat] ?? 0}</td>
                  <td>
                    <div className="bar-track">
                      <div className="bar" style={{ width: `${(100 * n) / max}%` }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {Object.keys(ins.memory).length > 0 && (
            <>
              <h3 className="section-title mt-4">Memory operations</h3>
              <table className="data">
                <thead><tr><th>Kind . state space</th><th className="num">Count</th></tr></thead>
                <tbody>
                  {Object.entries(ins.memory).map(([k, n]) => (
                    <tr key={k}><td><code>{k}</code></td><td className="num">{n}</td></tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </section>

        <section className="panel">
          <h3 className="section-title">Findings</h3>
          {fn.findings.length === 0 ? (
            <p className="muted text-sm">No findings for this function.</p>
          ) : (
            <ul className="flex flex-col gap-3">
              {fn.findings.map((f) => (
                <li key={f.rule} className={`finding ${f.severity}`}>
                  <div className="font-medium">{f.title}</div>
                  <div className="muted text-xs mb-1"><code>{f.rule}</code> · {f.severity}</div>
                  <p className="text-sm">{f.detail}</p>
                  {f.lines.length > 0 && (
                    <p className="muted text-xs mt-1">
                      PTX lines: {f.lines.slice(0, 16).join(", ")}{f.lines.length > 16 ? "…" : ""}
                    </p>
                  )}
                  {f.caveat && <p className="text-xs mt-1"><em>Caveat:</em> {f.caveat}</p>}
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>

      <section className="panel">
        <h3 className="section-title">Control-flow graph</h3>
        <p className="muted text-xs mb-3">
          Blocks in source order. Accent outline: loop header. Right-hand arcs: loop back edges.
          Shading: loop nesting depth. Hover a block for its PTX line range.
        </p>
        <div className="overflow-x-auto">
          <CfgGraph blocks={fn.cfg.blocks} headers={headers} />
        </div>
      </section>
    </div>
  );
}
