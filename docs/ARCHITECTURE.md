# KernelSight architecture

This document follows one input through the system and records the design
decisions at each stage. Module docstrings go into more detail; this is the map.

```
pipeline.analyze_source()
  ├─ toolchain.compile_cuda()      .cu → PTX            (NVRTC via ctypes)
  ├─ ptx.parse()                   PTX → IR             (lexer + recursive descent)
  ├─ toolchain.run_ptxas()         PTX → resource report (optional)
  └─ analysis.analyze()            IR → per-function analyses
        mix · cfg · registers · resources · findings
compare.compare(A, B)              two analyses → per-function diff
report.*_report()                  analyses/diffs → JSON dict → md / html / UI
cli.py · api.py                    entry points
```

## 1. Toolchain (`toolchain.py`)

- **NVRTC through `ctypes`.** NVRTC is a C library with a small API
  (`nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetPTX`, …). Calling it
  through `ctypes` avoids a dependency and a host C++ compiler, and shows the
  C ABI directly.
- **NVIDIA pip wheels** (`nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-nvcc-cu12`)
  provide NVRTC and `ptxas` without installing the CUDA Toolkit. Version 12.9
  is pinned so generated PTX can be loaded by drivers supporting CUDA ≥ 12.9.
- Discovery order: env override → wheels → `CUDA_PATH` → `PATH`.
  `kernelsight doctor` reports what was found.

## 2. Front end (`ptx/`)

- **Lexer** (`lexer.py`): one regular expression with named groups. A
  *word* token absorbs dots, so `ld.global.nc.f32`, `%tid.x` and
  `.shared::cta` each become one token; the parser splits opcodes on `.`.
  Tokens keep line numbers and source offsets.
- **Parser** (`parser.py`): hand-written recursive descent, one method per
  grammar rule. It is *tolerant* (unknown opcodes and operand forms are kept,
  as `Instruction` and `RawOperand`) but *strict about structure* (missing
  `;` or `}` raises `PTXSyntaxError` with a line). Debug sections
  (`.section .debug_*`) are kept as opaque directives.
- **IR** (`ir.py`): plain dataclasses. Instructions hold opcode, modifiers,
  predicate and typed operands (`Register`, `Immediate`, `Address`,
  `Vector`, …). Declarations keep their source text. Module items stay in
  source order because PTX requires declarations before use.
- **Printer** (`printer.py`): rebuilds instructions *from parsed fields*.
  That makes it a check on the parser: the round-trip test feeds printed
  PTX to `ptxas` and requires identical resource usage.

## 3. Analyses (`analysis/`)

| Module | What | Algorithm / source of truth |
|---|---|---|
| `opcodes.py` | category and state space per instruction | lookup table checked against the PTX ISA; misses go to `other` and are listed |
| `mix.py` | static instruction counts | one pass; "static" = per appearance, not per execution |
| `cfg.py` | basic blocks, edges, dominators, natural loops | leaders at labels and after branches; iterative dominator sets; back edge `T→H` iff `H dom T`; loop body by reverse walk from `T` to `H` |
| `registers.py` | declared virtual registers, def-use, peak liveness | backward data-flow to a fixed point; pressure = live-out ∪ defs per instruction, in 32-bit slots |
| `resources.py` | registers, spills, stack, smem from `ptxas -v` | regex over ptxas output; ground truth after register allocation |
| `findings.py` | rule-based observations | each rule cites PTX lines and a caveat; ptxas data can qualify a PTX signal |

Design choices worth knowing:

- **Why liveness and not declared registers?** PTX is in a near-SSA form with
  thousands of virtual registers; ptxas allocates physical registers. Peak
  liveness is the PTX-level quantity that relates to pressure; the report
  always shows ptxas's number next to it.
- **Self-loops.** A block that branches to itself is its own loop. The loop
  body walk must not continue past the header, a bug caught on real kernels
  and covered by a regression test.
- **Findings are conservative.** For example, `ld.local` in PTX does not
  guarantee local memory in machine code; when ptxas reports a 0-byte stack
  frame the finding says the accesses were optimised away.

## 4. Comparison (`compare.py`)

- Functions are matched by (mangled) name.
- Metric deltas: instruction mix, memory ops, mnemonics, CFG, liveness, ptxas.
- **Normalised diff.** Virtual registers and labels are renumbered by order
  of first use per function before `difflib`, so one inserted instruction
  doesn't renumber everything after it.
- Status `ptxas_only`: identical PTX but different ptxas results (e.g. the
  same PTX assembled for sm_75 and sm_89).

## 5. Reports (`report/`)

- One JSON-compatible dict per analysis/comparison with a schema id
  (`kernelsight.analysis/1`, `kernelsight.comparison/1`) and provenance
  (input sha256, NVRTC options/version, ptxas version/flags).
- No timestamps or absolute paths → identical input gives identical bytes;
  golden files in `tests/fixtures/golden` pin the output.
- Markdown, HTML and the web UI all render from the same dict.

## 6. Entry points

- `cli.py`: `analyze`, `compare`, `cfg`, `serve`, `doctor` (argparse).
- `api.py`: FastAPI. Compiler options from requests must match an allowlist,
  because they are passed to external tools.
- `frontend/`: Next.js static export served by `kernelsight serve`; types in
  `src/lib/api.ts` mirror the report schema.

## Testing strategy

| Layer | Tests |
|---|---|
| Lexer/parser | hand-written cases, error lines, prototype-bug regressions |
| Parser vs reality | round trip of the NVRTC corpus through `ptxas` |
| Analyses | small PTX functions with hand-computed CFGs, loops and liveness |
| Comparison | real option studies (fast-math, sm_75/sm_89, `-G`) |
| Reports | golden JSON, determinism, HTML escaping |
| CLI/API | commands, formats, error exits, unsafe-option rejection |

Toolchain-dependent tests are marked `toolchain` and skip automatically, so the
suite also runs (and runs in CI) without NVIDIA software.
