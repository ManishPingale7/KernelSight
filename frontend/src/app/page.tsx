"use client";

import { useEffect, useState } from "react";

import {
  api,
  download,
  type AnalysisReport,
  type ComparisonReport,
  type Example,
  type Format,
  type Health,
  type SourceInput,
} from "@/lib/api";
import { AnalysisView } from "@/components/AnalysisView";
import { CompareView } from "@/components/CompareView";
import { SourceEditor } from "@/components/SourceEditor";

type Mode = "analyze" | "compare";

const emptySource = (name = "kernel.cu"): SourceInput => ({
  source: "",
  kind: "cuda",
  name,
  arch: null,
  nvrtc_options: [],
  ptxas_options: [],
  use_ptxas: null,
});

export default function Home() {
  const [mode, setMode] = useState<Mode>("analyze");
  const [health, setHealth] = useState<Health | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [apiError, setApiError] = useState<string | null>(null);

  const [single, setSingle] = useState<SourceInput>(emptySource());
  const [pairA, setPairA] = useState<SourceInput>(emptySource());
  const [pairB, setPairB] = useState<SourceInput>({ ...emptySource(), nvrtc_options: ["--use_fast_math"] });

  const [analysis, setAnalysis] = useState<AnalysisReport | null>(null);
  const [comparison, setComparison] = useState<ComparisonReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Deep links: ?example=matmul.cu&run=analyze (or run=compare) preloads an
  // example and runs it, e.g. for sharing a demo or taking screenshots.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    Promise.all([api.health(), api.examples()])
      .then(async ([h, ex]) => {
        setHealth(h);
        setExamples(ex);
        const start = ex.find((e) => e.name === (params.get("example") ?? "math_heavy.cu")) ?? ex[0];
        if (!start) return;
        const src = { source: start.source, name: start.name, kind: "cuda" as const };
        const s = { ...emptySource(), ...src };
        const a = { ...emptySource(), ...src };
        const b = { ...emptySource(), ...src, nvrtc_options: ["--use_fast_math"] };
        setSingle(s);
        setPairA(a);
        setPairB(b);
        const run = params.get("run");
        if (run === "analyze") setAnalysis((await api.analyze(s)) as AnalysisReport);
        if (run === "compare") {
          setMode("compare");
          setComparison((await api.compare(a, b)) as ComparisonReport);
        }
      })
      .catch(() => setApiError("Cannot reach the KernelSight API. Start it with: kernelsight serve"));
  }, []);

  const toolchain = !!health?.toolchain.nvrtc.available && !!health?.toolchain.ptxas.available;

  async function run() {
    setBusy(true);
    setError(null);
    try {
      if (mode === "analyze") setAnalysis((await api.analyze(single)) as AnalysisReport);
      else setComparison((await api.compare(pairA, pairB)) as ComparisonReport);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function exportReport(format: Format) {
    try {
      const base = mode === "analyze" ? single.name : `${pairA.name}-vs-${pairB.name}`;
      const stem = base.replace(/\.[^.]+$/, "");
      if (format === "json") {
        const rep = mode === "analyze" ? analysis : comparison;
        download(`${stem}.json`, JSON.stringify(rep, null, 2) + "\n", "application/json");
        return;
      }
      const text = (mode === "analyze"
        ? await api.analyze(single, format)
        : await api.compare(pairA, pairB, format)) as string;
      download(`${stem}.${format}`, text, format === "html" ? "text/html" : "text/markdown");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const hasResult = mode === "analyze" ? !!analysis : !!comparison;
  const needsToolchain =
    mode === "analyze" ? single.kind === "cuda" : pairA.kind === "cuda" || pairB.kind === "cuda";

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-5 px-4 py-6">
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">KernelSight</h1>
          <p className="muted text-sm">PTX static analysis and compiler output comparison</p>
        </div>
        <div className="muted text-xs text-right">
          {health ? (
            <>
              v{health.version} · NVRTC {health.toolchain.nvrtc.version ?? "missing"} · ptxas{" "}
              {health.toolchain.ptxas.available ? "available" : "missing"}
            </>
          ) : (
            "connecting…"
          )}
        </div>
      </header>

      {apiError && <div className="alert">{apiError}</div>}

      <div className="segmented self-start" role="tablist" aria-label="Mode">
        {(["analyze", "compare"] as const).map((m) => (
          <button key={m} role="tab" aria-selected={mode === m} className={mode === m ? "active" : ""}
            onClick={() => setMode(m)}>
            {m === "analyze" ? "Analyze" : "Compare"}
          </button>
        ))}
      </div>

      {mode === "analyze" ? (
        <SourceEditor value={single} onChange={setSingle} examples={examples} toolchainAvailable={toolchain} />
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          <SourceEditor title="A" value={pairA} onChange={setPairA} examples={examples} toolchainAvailable={toolchain} />
          <SourceEditor title="B" value={pairB} onChange={setPairB} examples={examples} toolchainAvailable={toolchain} />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <button className="primary" onClick={run} disabled={busy || !health}>
          {busy ? "Running…" : mode === "analyze" ? "Analyze" : "Compare"}
        </button>
        {hasResult && (
          <>
            <span className="muted text-sm">Export:</span>
            {(["json", "md", "html"] as const).map((f) => (
              <button key={f} className="field" onClick={() => exportReport(f)}>
                {f.toUpperCase()}
              </button>
            ))}
          </>
        )}
        {needsToolchain && health && !health.toolchain.nvrtc.available && (
          <span className="text-sm warn-text">CUDA input needs NVRTC: pip install -e .[toolchain]</span>
        )}
      </div>

      {error && <pre className="alert whitespace-pre-wrap">{error}</pre>}

      {mode === "analyze" && analysis && <AnalysisView report={analysis} />}
      {mode === "compare" && comparison && <CompareView report={comparison} />}
    </main>
  );
}
