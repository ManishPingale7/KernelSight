# KernelSight analysis: histogram.cu

- Input: `histogram.cu` (cuda), sha256 `fff4e40c1f74371f…`
- NVRTC 12.9 options: `-arch=compute_89`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89`
- PTX ISA 8.8, target `sm_89`, 64-bit addresses

## `histogram256` (kernel, line 16)

| Metric | Value |
|---|---|
| Instructions (static) | 50 |
| …of which inside loops | 24 |
| Basic blocks / loops | 11 / 3 |
| Virtual registers declared | 47 |
| Peak live virtual registers (32-bit slots) | 13 (line 44) |
| ptxas registers | 10 |
| ptxas spill stores / loads (bytes) | 0 / 0 |
| ptxas stack frame / shared memory (bytes) | 0 / 1024 |

**Instruction mix**

| Category | Count | In loops |
|---|---|---|
| move / convert | 13 | 2 |
| integer arithmetic | 13 | 11 |
| memory | 8 | 5 |
| control flow | 8 | 3 |
| compare / select | 6 | 3 |
| synchronization | 2 | 0 |

**Memory operations** (kind.state space)

| Operation | Count |
|---|---|
| atomic.global | 1 |
| atomic.shared | 1 |
| load.global | 1 |
| load.param | 3 |
| load.shared | 1 |
| store.shared | 1 |

**Findings**

- **[info] 2 atomic operation(s) inside a loop** (`atomics-in-loop`)
  Atomics to the same address from many threads are serialised. Inside a loop this can dominate run time; per-block privatisation in shared memory or warp-level reduction first reduces contention.
  PTX lines: 69, 87
  _Caveat:_ Contention depends on the addresses touched at run time, which static analysis cannot see.

