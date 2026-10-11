"""Render report dicts as a single self-contained HTML page (no scripts,
no external resources), suitable for attaching to an issue or opening
offline."""

from __future__ import annotations

from html import escape

from kernelsight.report.markdown import CATEGORY_LABELS

_CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6b6b70; --line:#e3e3e6; --card:#f7f7f8;
        --accent:#2f6fde; --add:#e6f4ea; --del:#fdecea; --warn:#b15c00; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151517; --fg:#ececef; --muted:#9a9aa2; --line:#2c2c31; --card:#1d1d21;
          --accent:#6d9cf0; --add:#1c3324; --del:#3a1f1f; --warn:#f0a14a; } }
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 8px; } h2 { font-size: 19px; margin: 32px 0 8px; }
code, pre { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 13px; }
.muted { color: var(--muted); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
        padding: 16px; margin: 12px 0; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--line); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; }
.tile { border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; }
.tile b { display: block; font-size: 20px; font-variant-numeric: tabular-nums; }
.bar { height: 8px; background: var(--accent); border-radius: 4px; min-width: 2px; }
.finding { border-left: 3px solid var(--accent); padding: 4px 12px; margin: 10px 0; }
.finding.warning { border-color: var(--warn); }
pre.diff { margin: 0; white-space: pre; }
pre.diff span { display: block; }
pre.diff .add { background: var(--add); } pre.diff .del { background: var(--del); }
details > summary { cursor: pointer; margin: 8px 0; }
"""


def _page(title: str, body: list[str]) -> str:
    return (
        "<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
        + "\n".join(body)
        + "\n</main>\n</body>\n</html>\n"
    )


def _table(headers: list[str], rows: list[list], numeric_from: int = 1) -> str:
    def cell(tag, i, v):
        cls = " class='num'" if i >= numeric_from else ""
        return f"<{tag}{cls}>{escape('' if v is None else str(v))}</{tag}>"

    head = "".join(cell("th", i, h) for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell("td", i, v) for i, v in enumerate(r)) + "</tr>" for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _provenance(p: dict) -> str:
    items = []
    if inp := p.get("input"):
        name, kind, digest = escape(inp["name"]), escape(inp.get("kind", "?")), inp["sha256"][:16]
        items.append(f"Input <code>{name}</code> ({kind}), sha256 <code>{digest}…</code>")
    if nv := p.get("nvrtc"):
        options = escape(" ".join(nv["options"]))
        items.append(f"NVRTC {escape(nv['version'])}: <code>{options}</code>")
    if px := p.get("ptxas"):
        flags = escape(" ".join([f"-arch={px['arch']}", *px["options"]]))
        items.append(f"ptxas {escape(px['version'] or '')}, <code>{flags}</code>")
    else:
        items.append("ptxas not run: resource usage unavailable")
    return "<ul class='muted'>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def analysis_html(report: dict) -> str:
    name = report["provenance"].get("input", {}).get("name", "PTX")
    mod = report["module"]
    body = [
        f"<h1>KernelSight analysis: {escape(name)}</h1>",
        _provenance(report["provenance"]),
        f"<p class='muted'>PTX ISA {escape(str(mod['version']))}, target "
        f"<code>{escape(', '.join(mod['target']))}</code>, {mod['address_size']}-bit</p>",
    ]
    for fn in report["functions"]:
        body.append(_function_html(fn))
    return _page(f"KernelSight: {name}", body)


def _function_html(fn: dict) -> str:
    ins, loops, regs, px = fn["instructions"], fn["loops"], fn["registers"], fn["ptxas"]
    kind = "kernel" if fn["kind"] == "entry" else "device function"
    tiles = [
        ("Instructions", ins["total"]),
        ("In loops", loops["instructions_in_loops"]),
        ("Blocks / loops", f"{len(fn['cfg']['blocks'])} / {loops['count']}"),
        ("Virtual regs declared", regs["declared"]),
        ("Peak live (32-bit slots)", regs["peak_live_slots"]),
    ]
    if px:
        tiles += [
            ("ptxas registers", px["registers"]),
            ("Spills st / ld (B)", f"{px['spill_stores']} / {px['spill_loads']}"),
            ("Shared mem (B)", px["shared_mem"]),
        ]
    out = [
        f"<h2><code>{escape(fn['name'])}</code> <span class='muted'>{kind}, line "
        f"{fn['line']}</span></h2>",
        "<div class='tiles'>"
        + "".join(
            f"<div class='tile'><span class='muted'>{escape(k)}</span><b>{escape(str(v))}</b></div>"
            for k, v in tiles
        )
        + "</div>",
    ]

    total = max(ins["total"], 1)
    rows = ""
    for c, n in ins["by_category"].items():
        bar = f"<div class='bar' style='width:{100 * n / total:.1f}%'></div>"
        rows += (
            f"<tr><td>{escape(CATEGORY_LABELS.get(c, c))}</td><td class='num'>{n}</td>"
            f"<td class='num'>{loops['by_category_in_loops'].get(c, 0)}</td>"
            f"<td style='width:40%'>{bar}</td></tr>"
        )
    out.append(
        "<div class='card'><table><thead><tr><th>Category</th><th class='num'>Count</th>"
        f"<th class='num'>In loops</th><th></th></tr></thead><tbody>{rows}</tbody></table></div>"
    )
    if ins["memory"]:
        out.append(
            "<div class='card'>"
            + _table(["Memory operation", "Count"], list(map(list, ins["memory"].items())))
            + "</div>"
        )

    for f in fn["findings"]:
        lines = ", ".join(map(str, f["lines"][:20])) + ("…" if len(f["lines"]) > 20 else "")
        out.append(
            f"<div class='finding {escape(f['severity'])}'><b>{escape(f['title'])}</b> "
            f"<span class='muted'>({escape(f['rule'])})</span><br>{escape(f['detail'])}"
            + (f"<br><span class='muted'>PTX lines: {lines}</span>" if lines else "")
            + (f"<br><i>Caveat:</i> {escape(f['caveat'])}" if f["caveat"] else "")
            + "</div>"
        )

    blocks = [
        [
            b["name"],
            "–".join(map(str, b["lines"])) if b["lines"] else "",
            b["instructions"],
            b["loop_depth"],
            ", ".join(b["successors"]),
        ]
        for b in fn["cfg"]["blocks"]
    ]
    out.append(
        "<details><summary>Control-flow graph: basic blocks</summary><div class='card'>"
        + _table(["Block", "Lines", "Instructions", "Loop depth", "Successors"], blocks, 2)
        + "</div></details>"
    )
    return "\n".join(out)


def comparison_html(report: dict) -> str:
    a, b = report["a"]["label"], report["b"]["label"]
    body = [f"<h1>KernelSight comparison: {escape(a)} → {escape(b)}</h1>"]
    for side in ("a", "b"):
        body.append(f"<p><b>{escape(report[side]['label'])}</b></p>")
        body.append(_provenance(report[side]["provenance"]))
    for key, (va, vb) in report["header"].items():
        before, after = escape(str(va)), escape(str(vb))
        change = f"<code>{before}</code> → <code>{after}</code>"
        body.append(f"<p>Module <code>{escape(key)}</code>: {change}</p>")
    for fd in report["functions"]:
        body.append(_function_diff_html(fd, a, b))
    return _page(f"KernelSight: {a} vs {b}", body)


def _function_diff_html(fd: dict, a: str, b: str) -> str:
    status = escape(fd["status"].replace("_", " "))
    out = [f"<h2><code>{escape(fd['name'])}</code> <span class='muted'>{status}</span></h2>"]
    if fd["status"] in ("only_in_a", "only_in_b"):
        return "\n".join(out)
    rows = []
    for k, (va, vb) in fd["metrics"].items():
        delta = "" if va is None or vb is None else f"{vb - va:+d}" if vb != va else "0"
        rows.append([k.replace("_", " "), va, vb, delta])
    out.append("<div class='card'>" + _table(["Metric", a, b, "Δ"], rows) + "</div>")
    if fd["changed_mnemonics"]:
        out.append(
            "<div class='card'>"
            + _table(
                ["Changed mnemonic", a, b], [[k, *v] for k, v in fd["changed_mnemonics"].items()]
            )
            + "</div>"
        )
    if fd["findings_added"] or fd["findings_removed"]:
        out.append(
            f"<p>Findings added: {escape(', '.join(fd['findings_added']) or 'none')}; "
            f"removed: {escape(', '.join(fd['findings_removed']) or 'none')}</p>"
        )
    if fd["diff"]:
        lines = []
        for ln in fd["diff"]:
            cls = ""
            if ln.startswith("+") and not ln.startswith("+++"):
                cls = "add"
            elif ln.startswith("-") and not ln.startswith("---"):
                cls = "del"
            lines.append(f"<span class='{cls}'>{escape(ln) or ' '}</span>")
        out.append(
            "<details><summary>Normalised instruction diff</summary><div class='card'>"
            f"<pre class='diff'>{''.join(lines)}</pre></div></details>"
        )
    return "\n".join(out)
