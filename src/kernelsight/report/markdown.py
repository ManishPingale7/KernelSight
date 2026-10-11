"""Render report dicts as Markdown (also the CLI's default terminal output)."""

from __future__ import annotations

CATEGORY_LABELS = {
    "memory": "memory",
    "fp_arith": "float arithmetic",
    "int_arith": "integer arithmetic",
    "special_fn": "special functions (SFU)",
    "logic": "logic / shift",
    "compare": "compare / select",
    "move_convert": "move / convert",
    "control": "control flow",
    "sync": "synchronization",
    "warp": "warp-level",
    "tensor": "tensor core",
    "other": "other",
}


def _table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if c is None else str(c) for c in row) + " |" for row in rows]
    return out


def _provenance_lines(p: dict) -> list[str]:
    lines = []
    if inp := p.get("input"):
        lines.append(
            f"- Input: `{inp['name']}` ({inp.get('kind', '?')}), sha256 `{inp['sha256'][:16]}…`"
        )
    if nv := p.get("nvrtc"):
        lines.append(f"- NVRTC {nv['version']} options: `{' '.join(nv['options'])}`")
    if px := p.get("ptxas"):
        opts = " ".join(px["options"])
        lines.append(f"- ptxas: {px['version']}, `-arch={px['arch']}{' ' + opts if opts else ''}`")
    else:
        lines.append("- ptxas: not run (resource usage unavailable)")
    return lines


def analysis_markdown(report: dict) -> str:
    mod = report["module"]
    out = [
        f"# KernelSight analysis: {report['provenance'].get('input', {}).get('name', 'PTX')}",
        "",
        *_provenance_lines(report["provenance"]),
        f"- PTX ISA {mod['version']}, target `{', '.join(mod['target'])}`, "
        f"{mod['address_size']}-bit addresses",
        "",
    ]
    for fn in report["functions"]:
        out += _function_md(fn)
    return "\n".join(out) + "\n"


def _function_md(fn: dict) -> list[str]:
    ins, loops, regs, px = fn["instructions"], fn["loops"], fn["registers"], fn["ptxas"]
    kind = "kernel" if fn["kind"] == "entry" else "device function"
    out = [f"## `{fn['name']}` ({kind}, line {fn['line']})", ""]
    rows = [
        ["Instructions (static)", ins["total"]],
        ["…of which inside loops", loops["instructions_in_loops"]],
        ["Basic blocks / loops", f"{len(fn['cfg']['blocks'])} / {loops['count']}"],
        ["Virtual registers declared", regs["declared"]],
        [
            "Peak live virtual registers (32-bit slots)",
            f"{regs['peak_live_slots']} (line {regs['peak_live_line']})",
        ],
    ]
    if px:
        rows += [
            ["ptxas registers", px["registers"]],
            ["ptxas spill stores / loads (bytes)", f"{px['spill_stores']} / {px['spill_loads']}"],
            [
                "ptxas stack frame / shared memory (bytes)",
                f"{px['stack_frame']} / {px['shared_mem']}",
            ],
        ]
    out += _table(["Metric", "Value"], rows) + [""]

    out += ["**Instruction mix**", ""]
    out += _table(
        ["Category", "Count", "In loops"],
        [
            [CATEGORY_LABELS.get(c, c), n, loops["by_category_in_loops"].get(c, 0)]
            for c, n in ins["by_category"].items()
        ],
    )
    out.append("")
    if ins["memory"]:
        out += ["**Memory operations** (kind.state space)", ""]
        out += _table(["Operation", "Count"], [[k, v] for k, v in ins["memory"].items()]) + [""]

    if fn["findings"]:
        out += ["**Findings**", ""]
        for f in fn["findings"]:
            lines = ", ".join(str(n) for n in f["lines"][:12])
            more = "…" if len(f["lines"]) > 12 else ""
            out.append(f"- **[{f['severity']}] {f['title']}** (`{f['rule']}`)")
            out.append(f"  {f['detail']}")
            if lines:
                out.append(f"  PTX lines: {lines}{more}")
            if f["caveat"]:
                out.append(f"  _Caveat:_ {f['caveat']}")
        out.append("")
    else:
        out += ["_No findings._", ""]
    return out


def comparison_markdown(report: dict) -> str:
    a, b = report["a"]["label"], report["b"]["label"]
    out = [f"# KernelSight comparison: {a} → {b}", ""]
    for side in ("a", "b"):
        out.append(f"**{report[side]['label']}**")
        out += _provenance_lines(report[side]["provenance"]) + [""]
    for key, (va, vb) in report["header"].items():
        out.append(f"- Module `{key}`: `{va}` → `{vb}`")
    s = report["summary"]
    out += [
        f"- Functions: {len(s['changed'])} changed, {len(s['identical'])} identical, "
        f"{len(s['only_in_a'])} only in {a}, {len(s['only_in_b'])} only in {b}",
        "",
    ]
    for fd in report["functions"]:
        out += _function_diff_md(fd, a, b)
    return "\n".join(out) + "\n"


def _function_diff_md(fd: dict, a: str, b: str) -> list[str]:
    out = [f"## `{fd['name']}`: {fd['status'].replace('_', ' ')}", ""]
    if fd["status"] in ("only_in_a", "only_in_b"):
        return out
    rows = [[k.replace("_", " "), va, vb, _delta(va, vb)] for k, (va, vb) in fd["metrics"].items()]
    out += _table(["Metric", a, b, "Δ"], rows) + [""]
    if fd["changed_mnemonics"]:
        out += ["**Changed instructions** (count per mnemonic)", ""]
        out += _table(
            ["Mnemonic", a, b],
            [[f"`{k}`", va, vb] for k, (va, vb) in fd["changed_mnemonics"].items()],
        )
        out.append("")
    if fd["findings_added"] or fd["findings_removed"]:
        added = ", ".join(fd["findings_added"]) or "none"
        removed = ", ".join(fd["findings_removed"]) or "none"
        out += [f"Findings added: {added}; removed: {removed}", ""]
    if fd["diff"]:
        out += ["<details><summary>Normalised instruction diff</summary>", "", "```diff"]
        out += fd["diff"] + ["```", "", "</details>", ""]
    return out


def _delta(va, vb) -> str:
    if va is None or vb is None:
        return ""
    d = vb - va
    return f"{d:+d}" if d else "0"
