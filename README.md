# KernelSight 

> PTX-level static analysis and ML-based performance prediction for CUDA kernels — before they ever run on hardware.

---

## Overview

GPU kernel optimization is traditionally a trial-and-error process: write, run, profile, repeat. KernelSight breaks this cycle by analyzing CUDA kernels at the PTX instruction level and predicting performance bottlenecks **prior to runtime execution**.

By treating PTX as a structured feature space — instruction mix, register pressure, memory access patterns — KernelSight trains a machine learning model to identify whether a kernel is memory-bound, compute-bound, or suffering from warp divergence, and surfaces actionable optimization suggestions through an interactive dashboard.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        KernelSight                          │
└─────────────────────────────────────────────────────────────┘

  Input Layer                Analysis Layer          Output Layer
  ───────────                ──────────────          ────────────

  ┌──────────┐   nvcc         ┌────────────┐         ┌──────────────────┐
  │ .cu file │ ──────────────▶│ PTX Parser │         │  Dashboard       │
  │  (CUDA   │  --ptx flag    │            │         │                  │
  │  kernel) │               │ • ld/st cnt │         │  Execution time  │
  └──────────┘               │ • mul/add   │         │  Bottleneck type │
                             │ • branches  │         │  Instruction mix │
                             │ • registers │         │  Suggestions     │
                             └─────┬──────┘         └────────┬─────────┘
                                   │                          │
                                   ▼                          │
                          ┌────────────────┐                  │
                          │ Feature Vector │                  │
                          │                │                  │
                          │ [mem_ratio,    │                  │
                          │  reg_count,    │                  │
                          │  branch_cnt,   │                  │
                          │  arith_inten,  │                  │
                          │  total_instr]  │                  │
                          └───────┬────────┘                  │
                                  │                           │
                                  ▼                           │
                          ┌───────────────┐                   │
                          │  ML Model     │                   │
                          │               │                   │
                          │ RandomForest  │ ──────────────────┘
                          │ trained on    │
                          │ 200+ kernels  │
                          └───────────────┘
```

---

## PTX Feature Extraction

PTX (Parallel Thread Execution) is NVIDIA's hardware-agnostic intermediate representation — the layer between CUDA C++ and actual GPU machine code. KernelSight treats PTX as a structured signal.

```
PTX Instruction         Feature Extracted         Performance Signal
───────────────         ─────────────────         ──────────────────
ld.global.f32           Memory load count         Memory bandwidth pressure
st.global.f32           Memory store count        Write bottleneck indicator
mul.f32 / mad.f32       Compute instruction cnt   Arithmetic intensity
bra / @p setp           Branch instruction cnt    Warp divergence risk
.reg .f32 %f<N>         Register declaration      Register pressure / spill risk
bar.sync                Synchronization points    Thread barrier overhead
```

**Arithmetic Intensity** — the ratio of compute instructions to memory instructions — is the primary predictor of whether a kernel is memory-bound or compute-bound. This is a well-established metric in GPU performance literature (Roofline Model).

---

## ML Pipeline

```
Data Collection
───────────────
  NVIDIA CUDA Samples          →  compile to PTX
  Rodinia Benchmark Suite      →  profile with NVIDIA Nsight
  Custom synthetic kernels     →  measure cudaEvent_t timing
                                        │
                                        ▼
                               [features, execution_time]
                               CSV dataset, ~200+ kernels

Feature Engineering
───────────────────
  Raw PTX counts → normalized ratios
  memory_ratio      = mem_ops / total_ops
  arithmetic_intens = compute_ops / mem_ops
  branch_density    = branch_ops / total_ops
  register_pressure = declared_regs / warp_size

Model Training
──────────────
  Algorithm  :  Random Forest Regressor / Classifier
  Target     :  Execution time (regression)
                Bottleneck type (classification)
  Validation :  5-fold cross validation
  Metrics    :  MAE, R² for regression
                F1-score for bottleneck classification
```

---

## Bottleneck Categories

| Type | Primary Signal | Description |
|------|---------------|-------------|
| Memory-Bound | High `mem_ratio` | Kernel spends more time waiting for data than computing |
| Compute-Bound | High `arithmetic_intensity` | GPU cores are the bottleneck, not memory bandwidth |
| Warp Divergence | High `branch_density` | Threads within a warp take different execution paths |
| Register Spill | High `register_pressure` | Register overflow into slow local memory |

---

## Project Status

| Component | Status |
|-----------|--------|
| PTX Parser | In Development |
| Feature Extractor | In Development |
| Dataset Collection | In Development |
| ML Model | In Development |
| Streamlit Dashboard | Planned |

---

## Tech Stack

- **Compilation** — NVIDIA nvcc, CUDA Toolkit
- **PTX Analysis** — Python, regex-based instruction parser
- **ML** — scikit-learn, NumPy, pandas
- **Visualization** — Streamlit, Matplotlib
- **Benchmarking** — NVIDIA Nsight Systems, cudaEvent_t profiling

---

## Background

This project is motivated by the Roofline performance model — a visual framework that characterizes kernel performance relative to hardware limits. By predicting where a kernel falls on the roofline (memory-bound vs compute-bound) statically from PTX, KernelSight enables faster optimization iteration without repeated GPU profiling runs.

---

## References

- NVIDIA PTX ISA Documentation
- Roofline: An Insightful Visual Performance Model — Williams et al.
- NVIDIA CUDA C++ Best Practices Guide
- Rodinia: A Benchmark Suite for Heterogeneous Computing — Che et al.

---

*Developed as part of research into compiler-level GPU performance tooling.*
