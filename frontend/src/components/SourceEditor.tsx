"use client";

import type { Example, SourceInput } from "@/lib/api";

const ARCHES = ["sm_75", "sm_80", "sm_86", "sm_89", "sm_90"];

// NVRTC options offered as checkboxes. The API accepts only an allowlist.
const OPTION_TOGGLES: { flag: string; label: string; hint: string }[] = [
  { flag: "--use_fast_math", label: "Fast math", hint: "approximate div/sqrt/transcendentals" },
  { flag: "-G", label: "Debug (-G)", hint: "disables device optimisations" },
  { flag: "-lineinfo", label: "Line info", hint: "adds .loc source-line directives" },
];

interface Props {
  title?: string;
  value: SourceInput;
  onChange: (next: SourceInput) => void;
  examples: Example[];
  toolchainAvailable: boolean;
}

export function SourceEditor({ title, value, onChange, examples, toolchainAvailable }: Props) {
  const set = (patch: Partial<SourceInput>) => onChange({ ...value, ...patch });
  const toggleOption = (flag: string) =>
    set({
      nvrtc_options: value.nvrtc_options.includes(flag)
        ? value.nvrtc_options.filter((o) => o !== flag)
        : [...value.nvrtc_options, flag],
    });

  const loadFile = async (file: File) =>
    set({
      source: await file.text(),
      name: file.name,
      kind: file.name.endsWith(".ptx") ? "ptx" : "cuda",
    });

  return (
    <section className="panel flex flex-col gap-3">
      {title && <h3 className="text-sm font-semibold">{title}</h3>}

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <div className="segmented" role="radiogroup" aria-label="Input kind">
          {(["cuda", "ptx"] as const).map((k) => (
            <button
              key={k}
              role="radio"
              aria-checked={value.kind === k}
              className={value.kind === k ? "active" : ""}
              onClick={() => set({ kind: k })}
            >
              {k === "cuda" ? "CUDA C++" : "PTX"}
            </button>
          ))}
        </div>

        <select
          className="field"
          aria-label="Load example"
          value=""
          onChange={(e) => {
            const ex = examples.find((x) => x.name === e.target.value);
            if (ex) set({ source: ex.source, name: ex.name, kind: "cuda" });
          }}
        >
          <option value="">Load example…</option>
          {examples.map((ex) => (
            <option key={ex.name} value={ex.name}>
              {ex.name}
            </option>
          ))}
        </select>

        <label className="field cursor-pointer">
          Open file…
          <input
            type="file"
            accept=".cu,.cuh,.ptx"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && loadFile(e.target.files[0])}
          />
        </label>

        <span className="muted truncate">{value.name}</span>
      </div>

      <textarea
        className="code-input"
        spellCheck={false}
        value={value.source}
        onChange={(e) => set({ source: e.target.value })}
        placeholder={value.kind === "cuda" ? "Paste CUDA C++ kernels…" : "Paste PTX…"}
        aria-label="Source code"
      />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
        <label className="flex items-center gap-2">
          Target
          <select
            className="field"
            value={value.arch ?? ""}
            onChange={(e) => set({ arch: e.target.value || null })}
          >
            <option value="">{value.kind === "ptx" ? "from PTX" : "sm_89 (default)"}</option>
            {ARCHES.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>

        {value.kind === "cuda" &&
          OPTION_TOGGLES.map((o) => (
            <label key={o.flag} className="flex items-center gap-1.5" title={o.hint}>
              <input
                type="checkbox"
                checked={value.nvrtc_options.includes(o.flag)}
                onChange={() => toggleOption(o.flag)}
              />
              {o.label}
            </label>
          ))}

        <label
          className="flex items-center gap-1.5"
          title={toolchainAvailable ? "real register/spill numbers" : "ptxas not installed"}
        >
          <input
            type="checkbox"
            disabled={!toolchainAvailable}
            checked={toolchainAvailable && value.use_ptxas !== false}
            onChange={(e) => set({ use_ptxas: e.target.checked ? null : false })}
          />
          Run ptxas
        </label>
      </div>
    </section>
  );
}
