"""Command-line entry point: ``kernelsight <command>``.

Commands
--------
analyze FILE           analyse a .ptx file, or a .cu file compiled with NVRTC
compare A B            compare two inputs (or one .cu file under two option sets)
cfg FILE -f NAME       print one function's control-flow graph as Graphviz DOT
doctor                 check the optional NVIDIA toolchain

Compiler options that start with '-' must be attached with '=':
``kernelsight analyze k.cu --opt=--use_fast_math``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kernelsight import __version__, pipeline, report, toolchain
from kernelsight.analysis.cfg import to_dot
from kernelsight.compare import compare
from kernelsight.ptx import PTXSyntaxError
from kernelsight.report import html, markdown

FORMATS = ("md", "json", "html")


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Report which optional tools are available."""
    print(f"kernelsight {__version__} (Python {sys.version.split()[0]})")
    for status in (toolchain.nvrtc_status(), toolchain.ptxas_status()):
        if status.available:
            print(f"  {status.name:6} OK       {status.version}  [{status.path}]")
        else:
            where = f"  [{status.path}]" if status.path else ""
            print(f"  {status.name:6} MISSING  {status.error}{where}")
    print("PTX analysis works without these; compiling .cu files and ptxas")
    print("resource usage need them: pip install -e .[toolchain]")
    return 0


def _ptxas_choice(args: argparse.Namespace) -> bool | None:
    return False if args.no_ptxas else None  # None = use ptxas if installed


def _write(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {output}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def _cmd_analyze(args: argparse.Namespace) -> int:
    result = pipeline.analyze_file(
        args.input,
        nvrtc_options=args.opt,
        arch=args.arch,
        use_ptxas=_ptxas_choice(args),
        ptxas_options=args.ptxas_opt,
    )
    rep = report.analysis_report(result.analysis, result.provenance)
    render = {"json": report.to_json, "md": markdown.analysis_markdown, "html": html.analysis_html}
    _write(render[args.format](rep), args.output)
    return 0


def _label(path: str, opts: list[str], arch: str | None) -> str:
    extra = " ".join([*(f"--arch={arch}" for _ in [0] if arch), *opts])
    return f"{Path(path).name}{' ' + extra if extra else ''}"


def _cmd_compare(args: argparse.Namespace) -> int:
    path_b = args.input_b or args.input_a
    side = []
    for path, opts, arch in (
        (args.input_a, args.a_opt, args.a_arch or args.arch),
        (path_b, args.b_opt, args.b_arch or args.arch),
    ):
        result = pipeline.analyze_file(
            path, nvrtc_options=opts, arch=arch, use_ptxas=_ptxas_choice(args)
        )
        side.append((result, _label(path, opts, arch)))
    (ra, la), (rb, lb) = side
    if la == lb:
        la, lb = f"A: {la}", f"B: {lb}"
    diff = compare(ra.analysis, rb.analysis, la, lb)
    rep = report.comparison_report(diff, ra.provenance, rb.provenance)
    render = {
        "json": report.to_json,
        "md": markdown.comparison_markdown,
        "html": html.comparison_html,
    }
    _write(render[args.format](rep), args.output)
    return 0


def _cmd_cfg(args: argparse.Namespace) -> int:
    result = pipeline.analyze_file(
        args.input, nvrtc_options=args.opt, arch=args.arch, use_ptxas=False
    )
    names = [fa.name for fa in result.analysis.functions]
    if args.function not in names:
        print(
            f"error: no function {args.function!r}; available: {', '.join(names)}", file=sys.stderr
        )
        return 1
    fa = result.analysis.function(args.function)
    _write(to_dot(fa.cfg, fa.name), args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernelsight",
        description="PTX static analysis and compiler output comparison.",
        epilog="Attach compiler options that start with '-' using '=': --opt=--use_fast_math",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, formats: bool = True) -> None:
        p.add_argument("--arch", help="target GPU, e.g. sm_89 (default: from the PTX / sm_89)")
        p.add_argument("-o", "--output", help="write to a file instead of stdout")
        if formats:
            p.add_argument("--format", choices=FORMATS, default="md", help="output format")
            p.add_argument("--no-ptxas", action="store_true", help="skip ptxas resource usage")

    p = sub.add_parser("analyze", help="analyse a .ptx or .cu file")
    p.add_argument("input")
    p.add_argument("--opt", action="append", default=[], help="NVRTC option (repeatable)")
    p.add_argument("--ptxas-opt", action="append", default=[], help="ptxas option (repeatable)")
    common(p)
    p.set_defaults(func=_cmd_analyze)

    p = sub.add_parser(
        "compare",
        help="compare two inputs",
        description="Compare two .ptx/.cu inputs. Give one .cu file to compare it under "
        "two option sets, e.g.: compare k.cu --b-opt=--use_fast_math",
    )
    p.add_argument("input_a")
    p.add_argument("input_b", nargs="?", help="defaults to input_a")
    p.add_argument("--a-opt", action="append", default=[], help="NVRTC option for A")
    p.add_argument("--b-opt", action="append", default=[], help="NVRTC option for B")
    p.add_argument("--a-arch", help="target for A (overrides --arch)")
    p.add_argument("--b-arch", help="target for B (overrides --arch)")
    common(p)
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("cfg", help="print a function's CFG as Graphviz DOT")
    p.add_argument("input")
    p.add_argument("-f", "--function", required=True)
    p.add_argument("--opt", action="append", default=[], help="NVRTC option (repeatable)")
    common(p, formats=False)
    p.set_defaults(func=_cmd_cfg)

    p = sub.add_parser("doctor", help="check the optional NVIDIA toolchain")
    p.set_defaults(func=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to a legacy code page that cannot print '→'.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except PTXSyntaxError as exc:
        print(f"error: PTX syntax error at {exc}", file=sys.stderr)
    except toolchain.ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
