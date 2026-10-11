# KernelSight analysis: register_pressure.cu

- Input: `register_pressure.cu` (cuda), sha256 `2d82bfb16ec4b683…`
- NVRTC 12.9 options: `-arch=compute_89`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89 -maxrregcount=24`
- PTX ISA 8.8, target `sm_89`, 64-bit addresses

## `many_live_values` (kernel, line 15)

| Metric | Value |
|---|---|
| Instructions (static) | 302 |
| …of which inside loops | 0 |
| Basic blocks / loops | 3 / 0 |
| Virtual registers declared | 303 |
| Peak live virtual registers (32-bit slots) | 69 (line 164) |
| ptxas registers | 24 |
| ptxas spill stores / loads (bytes) | 416 / 576 |
| ptxas stack frame / shared memory (bytes) | 352 / 0 |

**Instruction mix**

| Category | Count | In loops |
|---|---|---|
| integer arithmetic | 130 | 0 |
| float arithmetic | 128 | 0 |
| memory | 36 | 0 |
| move / convert | 5 | 0 |
| control flow | 2 | 0 |
| compare / select | 1 | 0 |

**Memory operations** (kind.state space)

| Operation | Count |
|---|---|
| load.global | 32 |
| load.param | 3 |
| store.global | 1 |

**Findings**

- **[warning] ptxas spilled registers (416 B stored, 576 B loaded per thread)** (`register-spills`)
  With 24 registers per thread, ptxas could not keep all live values in registers and moved some to local memory. Reducing values live at the same time, or relaxing a register limit, removes spills.
  _Caveat:_ Spills are decided after PTX, so they have no PTX line numbers.

## `dynamic_local_index` (kernel, line 335)

| Metric | Value |
|---|---|
| Instructions (static) | 43 |
| …of which inside loops | 0 |
| Basic blocks / loops | 3 / 0 |
| Virtual registers declared | 42 |
| Peak live virtual registers (32-bit slots) | 12 (line 376) |
| ptxas registers | 12 |
| ptxas spill stores / loads (bytes) | 0 / 0 |
| ptxas stack frame / shared memory (bytes) | 0 / 0 |

**Instruction mix**

| Category | Count | In loops |
|---|---|---|
| float arithmetic | 15 | 0 |
| memory | 10 | 0 |
| move / convert | 7 | 0 |
| integer arithmetic | 6 | 0 |
| control flow | 2 | 0 |
| logic / shift | 2 | 0 |
| compare / select | 1 | 0 |

**Memory operations** (kind.state space)

| Operation | Count |
|---|---|
| load.global | 1 |
| load.local | 1 |
| load.param | 3 |
| store.global | 1 |
| store.local | 4 |

**Findings**

- **[info] 5 local-memory access(es) in PTX** (`local-memory`)
  Per-thread arrays indexed with run-time values, or too large for registers, are placed in local memory. Despite the name it resides in device memory (cached in L1/L2), so it is much slower than registers.
  PTX lines: 367, 372, 377, 382, 389
  _Caveat:_ ptxas reports a 0-byte stack frame for this function, so these accesses were optimised away.

