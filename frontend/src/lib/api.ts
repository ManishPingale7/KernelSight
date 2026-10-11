// Typed client for the KernelSight API (src/kernelsight/api.py).
// The types mirror the JSON reports built in src/kernelsight/report/__init__.py.

// In `npm run dev` the UI runs on :3000 and the API on :8000. In the built
// static UI, `kernelsight serve` hosts both on the same origin.
const API = process.env.NODE_ENV === "development" ? "http://127.0.0.1:8000" : "";

export type SourceKind = "ptx" | "cuda";

export interface SourceInput {
  source: string;
  kind: SourceKind;
  name: string;
  arch?: string | null;
  nvrtc_options: string[];
  ptxas_options: string[];
  use_ptxas: boolean | null;
}

export interface Provenance {
  input?: { name: string; kind: SourceKind; sha256: string };
  nvrtc?: { version: string; options: string[]; log: string | null };
  ptxas?: { version: string | null; arch: string; options: string[] };
  ptx_sha256?: string;
}

export interface Finding {
  rule: string;
  severity: "info" | "warning";
  title: string;
  detail: string;
  lines: number[];
  caveat: string | null;
}

export interface Block {
  id: number;
  name: string;
  lines: [number, number] | null;
  instructions: number;
  loop_depth: number;
  successors: string[];
}

export interface FunctionReport {
  name: string;
  kind: "entry" | "func";
  line: number;
  params: string[];
  instructions: {
    total: number;
    predicated: number;
    by_category: Record<string, number>;
    memory: Record<string, number>;
    uncategorized: string[];
  };
  loops: {
    count: number;
    max_depth: number;
    instructions_in_loops: number;
    by_category_in_loops: Record<string, number>;
  };
  cfg: {
    blocks: Block[];
    loops: { header: string; depth: number; blocks: string[] }[];
    unreachable: string[];
  };
  registers: {
    declared: number;
    declared_by_type: Record<string, number>;
    declared_slots: number;
    peak_live_slots: number;
    peak_live_line: number | null;
    peak_live_predicates: number;
  };
  ptxas: {
    registers: number | null;
    spill_stores: number;
    spill_loads: number;
    stack_frame: number;
    shared_mem: number;
    barriers: number;
  } | null;
  findings: Finding[];
}

export interface AnalysisReport {
  schema: string;
  tool: { name: string; version: string };
  provenance: Provenance;
  module: { version: string | null; target: string[]; address_size: number | null };
  functions: FunctionReport[];
}

type Pair = [number | null, number | null];

export interface FunctionDiff {
  name: string;
  status: "changed" | "ptxas_only" | "identical" | "only_in_a" | "only_in_b";
  metrics: Record<string, Pair>;
  by_category: Record<string, Pair>;
  memory: Record<string, Pair>;
  changed_mnemonics: Record<string, Pair>;
  findings_added: string[];
  findings_removed: string[];
  diff: string[];
}

export interface ComparisonReport {
  schema: string;
  a: { label: string; provenance: Provenance };
  b: { label: string; provenance: Provenance };
  header: Record<string, [string, string]>;
  summary: Record<FunctionDiff["status"], string[]>;
  functions: FunctionDiff[];
}

export interface Health {
  status: string;
  version: string;
  toolchain: Record<"nvrtc" | "ptxas", { available: boolean; version: string | null }>;
}

export interface Example {
  name: string;
  source: string;
}

export type Format = "json" | "md" | "html";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      // keep the status line
    }
    throw new Error(message);
  }
  const type = res.headers.get("content-type") ?? "";
  return (type.includes("json") ? res.json() : res.text()) as Promise<T>;
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const api = {
  health: () => request<Health>("/api/health"),
  examples: () => request<Example[]>("/api/examples"),
  analyze: (src: SourceInput, format: Format = "json") =>
    post<AnalysisReport | string>(`/api/analyze?format=${format}`, src),
  compare: (a: SourceInput, b: SourceInput, format: Format = "json") =>
    post<ComparisonReport | string>(`/api/compare?format=${format}`, { a, b }),
};

export function download(filename: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
