import type { Provenance } from "@/lib/api";

export const CATEGORY_LABELS: Record<string, string> = {
  memory: "memory",
  fp_arith: "float arithmetic",
  int_arith: "integer arithmetic",
  special_fn: "special functions (SFU)",
  logic: "logic / shift",
  compare: "compare / select",
  move_convert: "move / convert",
  control: "control flow",
  sync: "synchronization",
  warp: "warp-level",
  tensor: "tensor core",
  other: "other",
};

export function Tile({
  label,
  value,
  hint,
  warn,
}: {
  label: string;
  value: string | number;
  hint?: string;
  warn?: boolean;
}) {
  return (
    <div className={`tile ${warn ? "warn" : ""}`}>
      <div className="muted text-xs">{label}</div>
      <div className="tile-value">{value}</div>
      {hint && <div className="muted text-xs">{hint}</div>}
    </div>
  );
}

export function ProvenanceList({ provenance: p }: { provenance: Provenance }) {
  return (
    <ul className="muted flex flex-col gap-0.5">
      {p.input && (
        <li>
          Input <code>{p.input.name}</code> ({p.input.kind}), sha256{" "}
          <code>{p.input.sha256.slice(0, 16)}…</code>
        </li>
      )}
      {p.nvrtc && (
        <li>
          NVRTC {p.nvrtc.version}: <code>{p.nvrtc.options.join(" ")}</code>
        </li>
      )}
      <li>
        {p.ptxas ? (
          <>
            ptxas {p.ptxas.version}: <code>{[`-arch=${p.ptxas.arch}`, ...p.ptxas.options].join(" ")}</code>
          </>
        ) : (
          "ptxas not run: resource usage unavailable"
        )}
      </li>
    </ul>
  );
}
