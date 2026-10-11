"""Regenerate the PTX test corpus in tests/fixtures/ptx from examples/kernels.

Usage:  python scripts/generate_fixtures.py

Each example kernel is compiled with NVRTC under several option sets
("variants"). The resulting PTX files are committed so the test suite runs
without the NVIDIA toolchain; manifest.json records exactly which compiler
version and options produced each file.
"""

from __future__ import annotations

import json
from pathlib import Path

from kernelsight import toolchain

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "examples" / "kernels"
OUT = ROOT / "tests" / "fixtures" / "ptx"

# variant name -> NVRTC options
VARIANTS: dict[str, list[str]] = {
    "sm89": ["-arch=compute_89"],
    "sm89_fastmath": ["-arch=compute_89", "--use_fast_math"],
}
# Extra variants for a few files, to exercise more PTX syntax.
EXTRA: dict[str, dict[str, list[str]]] = {
    "elementwise": {
        "sm89_lineinfo": ["-arch=compute_89", "-lineinfo"],  # .file / .loc directives
        "sm89_debug": ["-arch=compute_89", "-G"],  # unoptimized, .section debug info
    },
    "matmul": {"sm75": ["-arch=compute_75"]},
}


def _escaped_path(path: Path) -> str:
    """A path as it appears inside a PTX string literal (backslashes doubled)."""
    return str(path).replace("\\", "\\\\")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"nvrtc_version": toolchain.nvrtc_status().version, "files": {}}
    for cu in sorted(KERNELS.glob("*.cu")):
        variants = {**VARIANTS, **EXTRA.get(cu.stem, {})}
        for variant, options in variants.items():
            result = toolchain.compile_cuda(cu.read_text(), options, name=cu.name)
            # With -lineinfo/-G, NVRTC records the absolute path of the
            # current directory in `.file` directives. Replace it with the
            # repo-relative path so fixtures do not depend on the machine.
            ptx = result.ptx.replace(
                _escaped_path(Path.cwd() / cu.name), f"examples/kernels/{cu.name}"
            )
            out = OUT / f"{cu.stem}.{variant}.ptx"
            out.write_text(ptx, encoding="utf-8", newline="\n")
            manifest["files"][out.name] = {
                "source": f"examples/kernels/{cu.name}",
                "options": options,
            }
            print(f"{out.relative_to(ROOT)}  ({len(result.ptx.splitlines())} lines)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", newline="\n")


if __name__ == "__main__":
    main()
