"""Regenerate the example reports in examples/reports (needs the toolchain).

Usage:  python scripts/generate_reports.py

Each entry is a KernelSight CLI invocation; the reports are committed so
readers can browse real output without installing anything.
"""

from __future__ import annotations

from pathlib import Path

from kernelsight.cli import main

ROOT = Path(__file__).resolve().parent.parent
K = ROOT / "examples" / "kernels"
OUT = ROOT / "examples" / "reports"

STUDIES: dict[str, list[str]] = {
    # Single-kernel analyses
    "analyze-matmul": ["analyze", str(K / "matmul.cu")],
    "analyze-histogram": ["analyze", str(K / "histogram.cu")],
    "analyze-register-pressure-maxrreg24": [
        "analyze", str(K / "register_pressure.cu"), "--ptxas-opt=-maxrregcount=24",
    ],
    # Compiler-option comparisons
    "compare-math-heavy-fast-math": [
        "compare", str(K / "math_heavy.cu"), "--b-opt=--use_fast_math",
    ],
    "compare-matmul-sm75-vs-sm89": [
        "compare", str(K / "matmul.cu"), "--a-arch=sm_75", "--b-arch=sm_89",
    ],
    "compare-elementwise-debug": ["compare", str(K / "elementwise.cu"), "--b-opt=-G"],
}  # fmt: skip


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, args in STUDIES.items():
        for fmt in ("md", "html"):
            out = OUT / f"{name}.{fmt}"
            if main([*args, "--format", fmt, "-o", str(out)]) != 0:
                raise SystemExit(f"failed: {name}")


if __name__ == "__main__":
    run()
