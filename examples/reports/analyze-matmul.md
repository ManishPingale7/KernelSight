# KernelSight analysis: matmul.cu

- Input: `matmul.cu` (cuda), sha256 `17b349ea04d1c15d…`
- NVRTC 12.9 options: `-arch=compute_89`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89`
- PTX ISA 8.8, target `sm_89`, 64-bit addresses

## `matmul_naive` (kernel, line 17)

| Metric | Value |
|---|---|
| Instructions (static) | 81 |
| …of which inside loops | 30 |
| Basic blocks / loops | 10 / 2 |
| Virtual registers declared | 103 |
| Peak live virtual registers (32-bit slots) | 25 (line 84) |
| ptxas registers | 40 |
| ptxas spill stores / loads (bytes) | 0 / 0 |
| ptxas stack frame / shared memory (bytes) | 0 / 0 |

**Instruction mix**

| Category | Count | In loops |
|---|---|---|
| integer arithmetic | 30 | 11 |
| memory | 15 | 10 |
| move / convert | 15 | 0 |
| compare / select | 7 | 2 |
| control flow | 7 | 2 |
| float arithmetic | 5 | 5 |
| logic / shift | 2 | 0 |

**Memory operations** (kind.state space)

| Operation | Count |
|---|---|
| load.global | 10 |
| load.param | 4 |
| store.global | 1 |

_No findings._

## `matmul_tiled` (kernel, line 129)

| Metric | Value |
|---|---|
| Instructions (static) | 123 |
| …of which inside loops | 74 |
| Basic blocks / loops | 10 / 1 |
| Virtual registers declared | 133 |
| Peak live virtual registers (32-bit slots) | 26 (line 204) |
| ptxas registers | 38 |
| ptxas spill stores / loads (bytes) | 0 / 0 |
| ptxas stack frame / shared memory (bytes) | 0 / 2048 |

**Instruction mix**

| Category | Count | In loops |
|---|---|---|
| memory | 41 | 36 |
| integer arithmetic | 26 | 7 |
| float arithmetic | 16 | 16 |
| move / convert | 15 | 2 |
| logic / shift | 9 | 3 |
| compare / select | 8 | 5 |
| control flow | 6 | 3 |
| synchronization | 2 | 2 |

**Memory operations** (kind.state space)

| Operation | Count |
|---|---|
| load.global | 2 |
| load.param | 4 |
| load.shared | 32 |
| store.global | 1 |
| store.shared | 2 |

_No findings._

