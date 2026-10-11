"""Run the benchmark studies on the local GPU and write examples/benchmarks.

Usage:  python scripts/benchmark_study.py [--force]

Measurements are only meaningful on an otherwise idle GPU: the script checks
GPU utilisation and other GPU processes first and refuses to run if the GPU
is busy (--force records anyway, with the GPU state saved next to the
results). Close games, browsers with hardware acceleration, etc. and keep the
laptop on mains power.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernelsight import __version__, toolchain
from kernelsight.bench import STUDIES, W, run_workload
from kernelsight.gpu import Driver, busy_reason, gpu_state

OUT = Path(__file__).resolve().parent.parent / "examples" / "benchmarks"


def _speedup(a: dict, b: dict) -> float:
    return round(a["median_ms"] / b["median_ms"], 2)


def markdown(results: dict) -> str:
    env = results["environment"]
    out = [
        "# KernelSight benchmark studies",
        "",
        f"GPU: {env['gpu']} ({env['compute_capability']}), driver CUDA {env['driver_cuda']}; "
        f"code from NVRTC/ptxas {env['ptxas']}. Median of {env['repeats']} samples, each "
        "averaging a batch of 10 launches; IQR in parentheses. Every output was checked "
        "against a Python reference before timing.",
        "",
        "> These are observations on one laptop GPU, not a performance model. Laptop clocks "
        "vary with temperature and power; differences of a few percent are noise.",
        "",
    ]
    if env.get("forced"):
        out += [f"> **Warning:** recorded with --force while busy: {env['busy']}", ""]
    for study in results["studies"]:
        a, b = study["a"], study["b"]
        out += [f"## {study['title']}", "", f"_{study['question']}_", ""]
        out += [
            "| | " + a["workload"] + " | " + b["workload"] + " |",
            "|---|---|---|",
            f"| median time (ms) | {a['median_ms']} ({a['iqr_ms']}) | "
            f"{b['median_ms']} ({b['iqr_ms']}) |",
        ]
        if "effective_gb_per_s" in a:
            out.append(f"| effective bandwidth (GB/s) | {a['effective_gb_per_s']} | "
                       f"{b['effective_gb_per_s']} |")  # fmt: skip
        for key in ("instructions", "instructions_in_loops", "global_memory_ops",
                    "shared_memory_ops", "ptxas_registers", "spill_bytes"):  # fmt: skip
            out.append(f"| {key.replace('_', ' ')} | {a['static'][key]} | {b['static'][key]} |")
        out += ["", f"**B is {study['speedup_b_over_a']}× the speed of A.**", ""]
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="measure even if the GPU is busy")
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()

    state = gpu_state()
    busy = busy_reason(state)
    if busy and not args.force:
        raise SystemExit(f"Not measuring: {busy}. Close it and retry, or use --force.")

    driver = Driver()
    arch = driver.compute_capability()
    needed = sorted({w for _, a, b, _ in STUDIES for w in (a, b)})
    entries = {}
    for name in needed:
        print(f"running {name} ...", flush=True)
        entries[name] = run_workload(driver, W[name], arch, repeats=args.repeats)

    results = {
        "schema": "kernelsight.benchmarks/1",
        "tool": {"name": "kernelsight", "version": __version__},
        "environment": {
            "gpu": driver.name(),
            "compute_capability": arch,
            "driver_cuda": driver.driver_version(),
            "ptxas": toolchain.ptxas_status().version,
            "repeats": args.repeats,
            "gpu_state_before": state,
            "gpu_state_after": gpu_state(),
            "forced": bool(busy),
            "busy": busy,
        },
        "studies": [
            {
                "title": title,
                "question": question,
                "a": entries[a],
                "b": entries[b],
                "speedup_b_over_a": _speedup(entries[a], entries[b]),
            }
            for title, a, b, question in STUDIES
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n", newline="\n")
    (OUT / "results.md").write_text(markdown(results), encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'results.md'}")


if __name__ == "__main__":
    main()
