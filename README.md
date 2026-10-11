# KernelSight

**PTX static analysis and compiler output comparison for CUDA kernels.**

> **Status: under active rework.** An earlier version of this README described an
> ML performance predictor trained on 200+ kernels. That model and dataset were
> never built, so those claims have been removed. This README will be rewritten
> once the analysis pipeline below is implemented and tested.

## What KernelSight does (target)

PTX is the virtual instruction set the CUDA compiler emits before `ptxas`
assembles it into machine code for a specific GPU. KernelSight works at that
layer:

1. **Parse** PTX into a small, typed intermediate representation (modules,
   functions, instructions, predicates, operands).
2. **Analyze** it: instruction mix by memory space, control-flow graph and
   loops, virtual-register def-use, and, when `ptxas` is available, the real
   per-kernel register, spill and shared-memory usage.
3. **Compare** two compilations of the same kernel (e.g. `--use_fast_math`,
   loop unrolling, different target architectures) and explain the differences.
4. **Report** results as JSON, Markdown or HTML, recording the inputs and tool
   versions used so that a run can be reproduced.

The core works on PTX text alone and does not need a GPU or the CUDA Toolkit.

## Quick start (development)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"            # core + tests
pip install -e ".[toolchain]"      # optional: NVRTC + ptxas from NVIDIA pip wheels
kernelsight doctor                 # shows which optional tools were found
pytest
```

## Repository layout

| Path | Contents |
|------|----------|
| `src/kernelsight/` | Python library and CLI |
| `tests/` | pytest suite (toolchain-dependent tests skip automatically) |
| `backend/` | FastAPI service (being moved onto the library) |
| `frontend/` | Next.js UI (not yet implemented) |
