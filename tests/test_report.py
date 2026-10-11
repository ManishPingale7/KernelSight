"""Report tests: schema, determinism and golden files.

Golden files pin the exact JSON a report produces for a fixed input, so any
change in parsing, analysis or report layout shows up as a reviewable diff.
After an intentional change, regenerate them with:

    KERNELSIGHT_UPDATE_GOLDEN=1 pytest tests/test_report.py
"""

import json
import os
from html.parser import HTMLParser
from pathlib import Path

import pytest

from kernelsight import pipeline, report
from kernelsight.analysis import analyze
from kernelsight.compare import compare
from kernelsight.ptx import parse
from kernelsight.report import html, markdown

FIXTURES = Path(__file__).parent / "fixtures"
PTX = FIXTURES / "ptx"
GOLDEN = FIXTURES / "golden"


def analysis_for(name: str) -> dict:
    result = pipeline.analyze_file(PTX / name, use_ptxas=False)
    return report.analysis_report(result.analysis, result.provenance)


def comparison_for(name_a: str, name_b: str) -> dict:
    a = pipeline.analyze_file(PTX / name_a, use_ptxas=False)
    b = pipeline.analyze_file(PTX / name_b, use_ptxas=False)
    diff = compare(a.analysis, b.analysis, name_a, name_b)
    return report.comparison_report(diff, a.provenance, b.provenance)


def check_golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if os.environ.get("KERNELSIGHT_UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    assert text == path.read_text(encoding="utf-8"), f"{name} changed; see module docstring"


def test_analysis_golden():
    check_golden(
        "elementwise.sm89.analysis.json", report.to_json(analysis_for("elementwise.sm89.ptx"))
    )


def test_comparison_golden():
    rep = comparison_for("math_heavy.sm89.ptx", "math_heavy.sm89_fastmath.ptx")
    check_golden("math_heavy.fastmath.comparison.json", report.to_json(rep))


def test_report_is_deterministic():
    first = report.to_json(analysis_for("matmul.sm89.ptx"))
    second = report.to_json(analysis_for("matmul.sm89.ptx"))
    assert first == second


def test_schema_and_provenance():
    rep = analysis_for("histogram.sm89.ptx")
    assert rep["schema"] == report.ANALYSIS_SCHEMA
    assert rep["provenance"]["input"]["name"] == "histogram.sm89.ptx"
    assert len(rep["provenance"]["input"]["sha256"]) == 64
    (fn,) = rep["functions"]
    assert fn["loops"]["count"] == 3
    assert fn["ptxas"] is None  # not run
    json.loads(report.to_json(rep))  # valid JSON


@pytest.mark.parametrize("path", sorted(PTX.glob("*.ptx")), ids=lambda p: p.name)
def test_all_fixtures_render(path):
    rep = analysis_for(path.name)
    md = markdown.analysis_markdown(rep)
    page = html.analysis_html(rep)
    assert md.startswith("# KernelSight analysis")
    HTMLParser().feed(page)  # well-formed enough to parse
    assert all(f["name"] in md for f in rep["functions"])


def test_html_escapes_untrusted_text():
    module = parse(".entry k()\n{\nret;\n}")
    rep = report.analysis_report(
        analyze(module), {"input": {"name": "<script>x</script>", "sha256": "0" * 64}}
    )
    page = html.analysis_html(rep)
    assert "<script>" not in page and "&lt;script&gt;" in page


def test_comparison_renders_diff():
    rep = comparison_for("math_heavy.sm89.ptx", "math_heavy.sm89_fastmath.ptx")
    assert rep["summary"]["identical"] == ["polynomial"]
    assert "div.approx.ftz.f32" in markdown.comparison_markdown(rep)
    page = html.comparison_html(rep)
    assert "class='add'" in page and "class='del'" in page
