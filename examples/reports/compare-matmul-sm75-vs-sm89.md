# KernelSight comparison: matmul.cu --arch=sm_75 → matmul.cu --arch=sm_89

**matmul.cu --arch=sm_75**
- Input: `matmul.cu` (cuda), sha256 `17b349ea04d1c15d…`
- NVRTC 12.9 options: `-arch=compute_75`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_75`

**matmul.cu --arch=sm_89**
- Input: `matmul.cu` (cuda), sha256 `17b349ea04d1c15d…`
- NVRTC 12.9 options: `-arch=compute_89`
- ptxas: Cuda compilation tools, release 12.9, V12.9.86, `-arch=sm_89`

- Module `target`: `sm_75` → `sm_89`
- Functions: 0 changed, 2 with identical PTX but different ptxas results, 0 identical, 0 only in matmul.cu --arch=sm_75, 0 only in matmul.cu --arch=sm_89

## `matmul_naive`: identical PTX, different ptxas results

| Metric | matmul.cu --arch=sm_75 | matmul.cu --arch=sm_89 | Δ |
|---|---|---|---|
| instructions | 81 | 81 | 0 |
| instructions in loops | 30 | 30 | 0 |
| basic blocks | 10 | 10 | 0 |
| loops | 2 | 2 | 0 |
| virtual registers | 103 | 103 | 0 |
| peak live slots | 25 | 25 | 0 |
| ptxas registers | 49 | 40 | -9 |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 0 | 0 |
| ptxas shared mem | 0 | 0 | 0 |

## `matmul_tiled`: identical PTX, different ptxas results

| Metric | matmul.cu --arch=sm_75 | matmul.cu --arch=sm_89 | Δ |
|---|---|---|---|
| instructions | 123 | 123 | 0 |
| instructions in loops | 74 | 74 | 0 |
| basic blocks | 10 | 10 | 0 |
| loops | 1 | 1 | 0 |
| virtual registers | 133 | 133 | 0 |
| peak live slots | 26 | 26 | 0 |
| ptxas registers | 39 | 38 | -1 |
| ptxas spill stores | 0 | 0 | 0 |
| ptxas spill loads | 0 | 0 | 0 |
| ptxas stack frame | 0 | 0 | 0 |
| ptxas shared mem | 2048 | 2048 | 0 |

