"use client";

import { useState } from "react";

import type { ComparisonReport, FunctionDiff } from "@/lib/api";
import { ProvenanceList } from "@/components/common";

const STATUS_LABEL: Record<FunctionDiff["status"], string> = {
  changed: "changed",
  identical: "identical",
  only_in_a: "only in A",
  only_in_b: "only in B",
};

export function CompareView({ report }: { report: ComparisonReport }) {
  const firstChanged = Math.max(0, report.functions.findIndex((f) => f.status === "changed"));
  const [selected, setSelected] = useState(firstChanged);
  const fd = report.functions[Math.min(selected, report.functions.length - 1)];
  const a = report.a.label;
  const b = report.b.label;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 md:grid-cols-2 text-sm">
        {(["a", "b"] as const).map((side) => (
          <div key={side} className="panel">
            <div className="font-medium mb-1">
              {side.toUpperCase()}: {report[side].label}
            </div>
            <ProvenanceList provenance={report[side].provenance} />
          </div>
        ))}
      </div>

      {Object.entries(report.header).map(([k, [va, vb]]) => (
        <p key={k} className="text-sm">
          Module <code>{k}</code>: <code>{va}</code> → <code>{vb}</code>
        </p>
      ))}

      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Functions">
        {report.functions.map((f, i) => (
          <button
            key={f.name}
            role="tab"
            aria-selected={f === fd}
            className={`chip ${f === fd ? "active" : ""}`}
            onClick={() => setSelected(i)}
          >
            <code>{f.name}</code>
            <span className={`status ${f.status}`}>{STATUS_LABEL[f.status]}</span>
          </button>
        ))}
      </div>

      {fd && <FunctionDiffPanel key={fd.name} fd={fd} a={a} b={b} />}
    </div>
  );
}

function delta(va: number | null, vb: number | null) {
  if (va === null || vb === null) return "";
  const d = vb - va;
  return d === 0 ? "0" : d > 0 ? `+${d}` : `${d}`;
}

function FunctionDiffPanel({ fd, a, b }: { fd: FunctionDiff; a: string; b: string }) {
  if (fd.status === "only_in_a" || fd.status === "only_in_b") {
    return (
      <p className="panel text-sm">
        <code>{fd.name}</code> exists only in {fd.status === "only_in_a" ? a : b}.
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="panel">
          <h3 className="section-title">Metrics</h3>
          <table className="data">
            <thead><tr><th>Metric</th><th className="num">A</th><th className="num">B</th><th className="num">Δ</th></tr></thead>
            <tbody>
              {Object.entries(fd.metrics).map(([k, [va, vb]]) => (
                <tr key={k} className={va !== vb ? "changed-row" : ""}>
                  <td>{k.replaceAll("_", " ")}</td>
                  <td className="num">{va ?? "–"}</td>
                  <td className="num">{vb ?? "–"}</td>
                  <td className="num">{delta(va, vb)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {(fd.findings_added.length > 0 || fd.findings_removed.length > 0) && (
            <p className="text-sm mt-3">
              Findings added: <code>{fd.findings_added.join(", ") || "none"}</code>; removed:{" "}
              <code>{fd.findings_removed.join(", ") || "none"}</code>
            </p>
          )}
        </section>

        <section className="panel">
          <h3 className="section-title">Changed instructions</h3>
          {Object.keys(fd.changed_mnemonics).length === 0 ? (
            <p className="muted text-sm">Same instruction counts per mnemonic.</p>
          ) : (
            <div className="max-h-96 overflow-y-auto">
              <table className="data">
                <thead><tr><th>Mnemonic</th><th className="num">A</th><th className="num">B</th></tr></thead>
                <tbody>
                  {Object.entries(fd.changed_mnemonics).map(([m, [va, vb]]) => (
                    <tr key={m}><td><code>{m}</code></td><td className="num">{va}</td><td className="num">{vb}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      <section className="panel">
        <h3 className="section-title">Normalised instruction diff</h3>
        <p className="muted text-xs mb-2">
          Registers and labels are renumbered by first use in each version, so compiler
          renumbering does not show up as a change.
        </p>
        {fd.diff.length === 0 ? (
          <p className="muted text-sm">No differences.</p>
        ) : (
          <pre className="diff">
            {fd.diff.map((line, i) => {
              const cls =
                line.startsWith("+") && !line.startsWith("+++") ? "add"
                : line.startsWith("-") && !line.startsWith("---") ? "del"
                : line.startsWith("@@") ? "hunk" : "";
              return <span key={i} className={cls}>{line || " "}</span>;
            })}
          </pre>
        )}
      </section>
    </div>
  );
}
