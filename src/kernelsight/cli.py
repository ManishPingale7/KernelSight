"""Command-line entry point: ``kernelsight <command>``."""

from __future__ import annotations

import argparse
import sys

from kernelsight import __version__, toolchain


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernelsight",
        description="PTX static analysis and compiler output comparison.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check the optional NVIDIA toolchain")
    doctor.set_defaults(func=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
