"""Reports: one JSON-compatible dict per analysis or comparison.

All output formats (terminal text, Markdown, HTML, the web UI) are rendered
from these dicts, so they always agree. The dicts carry a schema identifier
and the provenance of the input, and contain no timestamps or absolute
paths, so the same input always produces byte-identical output.
"""

from __future__ import annotations

import json
from collections import Counter

from kernelsight import __version__
from kernelsight.analysis import FunctionAnalysis, ModuleAnalysis
from kernelsight.analysis.registers import slots
from kernelsight.compare import FunctionDiff, ModuleDiff

ANALYSIS_SCHEMA = "kernelsight.analysis/1"
COMPARISON_SCHEMA = "kernelsight.comparison/1"


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


def _tool() -> dict:
    return {"name": "kernelsight", "version": __version__}


# --------------------------------------------------------------------------
# Single analysis
# --------------------------------------------------------------------------


def analysis_report(analysis: ModuleAnalysis, provenance: dict | None = None) -> dict:
    m = analysis.module
    return {
        "schema": ANALYSIS_SCHEMA,
        "tool": _tool(),
        "provenance": provenance or {},
        "module": {
            "version": m.version,
            "target": list(m.target),
            "address_size": m.address_size,
        },
        "functions": [_function(fa) for fa in analysis.functions],
    }


def _function(fa: FunctionAnalysis) -> dict:
    fn, mix, cfg, regs = fa.function, fa.mix, fa.cfg, fa.registers
    names = {b.id: b.name for b in cfg.blocks}
    by_type = Counter(regs.declared.values())
    return {
        "name": fn.name,
        "kind": fn.kind,
        "line": fn.line,
        "params": [p.name for p in fn.params],
        "instructions": {
            "total": mix.total,
            "predicated": mix.predicated,
            "by_category": dict(mix.by_category.most_common()),
            "memory": {f"{k}.{s}": n for (k, s), n in sorted(mix.memory.items())},
            "uncategorized": sorted(mix.uncategorized),
        },
        "loops": {
            "count": len(cfg.loops),
            "max_depth": max((lp.depth for lp in cfg.loops), default=0),
            "instructions_in_loops": fa.loop_mix.total,
            "by_category_in_loops": dict(fa.loop_mix.by_category.most_common()),
        },
        "cfg": {
            "blocks": [
                {
                    "id": b.id,
                    "name": b.name,
                    "lines": list(b.lines) if b.lines else None,
                    "instructions": len(b.instructions),
                    "loop_depth": b.loop_depth,
                    "successors": [names[s] for s in b.successors],
                }
                for b in cfg.blocks
            ],
            "loops": [
                {
                    "header": names[lp.header],
                    "depth": lp.depth,
                    "blocks": sorted(names[i] for i in lp.blocks),
                }
                for lp in cfg.loops
            ],
            "unreachable": [names[i] for i in cfg.unreachable],
        },
        "registers": {
            "declared": regs.declared_count,
            "declared_by_type": dict(sorted(by_type.items())),
            "declared_slots": sum(slots(t) for t in regs.declared.values()),
            "peak_live_slots": regs.max_live_slots,
            "peak_live_line": regs.max_live_line,
            "peak_live_predicates": regs.max_live_predicates,
        },
        "ptxas": _resources(fa),
        "findings": [
            {
                "rule": f.rule,
                "severity": f.severity,
                "title": f.title,
                "detail": f.detail,
                "lines": f.lines,
                "caveat": f.caveat,
            }
            for f in fa.findings
        ],
    }


def _resources(fa: FunctionAnalysis) -> dict | None:
    r = fa.resources
    if r is None:
        return None
    return {
        "registers": r.registers,
        "spill_stores": r.spill_stores,
        "spill_loads": r.spill_loads,
        "stack_frame": r.stack_frame,
        "shared_mem": r.shared_mem,
        "barriers": r.barriers,
        "const_mem": {str(k): v for k, v in sorted(r.const_mem.items())},
    }


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def comparison_report(
    diff: ModuleDiff, provenance_a: dict | None = None, provenance_b: dict | None = None
) -> dict:
    return {
        "schema": COMPARISON_SCHEMA,
        "tool": _tool(),
        "a": {"label": diff.label_a, "provenance": provenance_a or {}},
        "b": {"label": diff.label_b, "provenance": provenance_b or {}},
        "header": {k: list(v) for k, v in diff.header.items()},
        "summary": {
            status: [f.name for f in diff.functions if f.status == status]
            for status in ("changed", "ptxas_only", "identical", "only_in_a", "only_in_b")
        },
        "functions": [_function_diff(fd) for fd in diff.functions],
    }


def _function_diff(fd: FunctionDiff) -> dict:
    return {
        "name": fd.name,
        "status": fd.status,
        "metrics": {k: list(v) for k, v in fd.metrics.items()},
        "by_category": {k: list(v) for k, v in fd.categories.items()},
        "memory": {k: list(v) for k, v in fd.memory.items()},
        "changed_mnemonics": {k: list(v) for k, v in fd.mnemonics.items()},
        "findings_added": fd.findings_added,
        "findings_removed": fd.findings_removed,
        "diff": fd.diff,
    }
