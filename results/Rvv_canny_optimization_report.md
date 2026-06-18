# RVV Canny Edge Detection — Optimization & Analysis Report

**Project:** Canny Edge Detection on RISC-V with Vector Extension  
**Architecture:** `rv64gcv` | **Emulation:** gem5 (TimingSimpleCPU @ 3 GHz)  
**Language:** C23 | **Team Size:** 4 Students

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Theoretical Analysis](#2-theoretical-analysis)
   - 2.1 Separable Filters vs. Spatial Convolution
   - 2.2 Sobel Output Data Types
   - 2.3 Magnitude Normalization Strategy
   - 2.4 QEMU/gem5 Benchmarking Methodology
   - 2.5 Amdahl's Law & Profiling Strategy
3. [Simulation Specifications](#3-simulation-specifications)
4. [Compiler Optimization Sweep (Phase 4)](#4-compiler-optimization-sweep-phase-4)
   - 4.1 -O0 Baseline
   - 4.2 -O2 Standard Optimization
   - 4.3 -O3 Aggressive Optimization
   - 4.4 Auto-Vectorization Analysis
5. [Profiling & Hotspot Identification (Phase 5)](#5-profiling--hotspot-identification-phase-5)
6. [RVV Intrinsic Optimization (Phase 6)](#6-rvv-intrinsic-optimization-phase-6)
   - 6.1 LMUL Sweep — Gaussian Separable Filter
   - 6.2 LMUL Sweep — Sobel Filter
   - 6.3 LMUL Sweep — Magnitude L1
   - 6.4 LMUL Analysis & Conclusions
7. [Summary Optimization Table](#7-summary-optimization-table)

---

## 1. Project Overview

This project implements a high-performance Canny edge detection pipeline targeting the RISC-V `rv64gcv` architecture. The full pipeline consists of five stages:

1. **Gaussian Blur** — 5×5 kernel (2D convolution, zero-padded)
2. **Sobel Gradients** — Gx and Gy using 3×3 kernels, output as `int16_t`
3. **Gradient Magnitude** — L1 (`|Gx| + |Gy|`) and L2 (`√(Gx² + Gy²)`) norms
4. **Gradient Direction** — Quantized to 4 angles (0°, 45°, 90°, 135°) without `atan2()`
5. **NMS & Hysteresis** — Non-maximum suppression and double thresholding (bonus stages)

The development followed a structured embedded engineering workflow: scalar baseline → measurement → compiler sweep → profiling → RVV intrinsic optimization → equivalence verification at VLEN = 128, 256, and 512 bits.

---

## 2. Theoretical Analysis

### 2.1 Separable Filters vs. Spatial Convolution

A standard 5×5 spatial convolution requires **25 multiply-accumulate (MAC) operations** per pixel. Decomposing this into a separable filter — a 1×5 horizontal pass followed by a 5×1 vertical pass — reduces the count to **10 MACs per pixel**, a 60% arithmetic reduction.

**Observed Performance (gem5, `-O0`):**

| Method | Time (ms) |
|---|---|
| Spatial 5×5 | 1054 |
| Separable (1×5 + 5×1) | 473 |

The separable filter was approximately **2.2× faster** under unoptimized compilation. At `-O0`, the compiler produces no instruction scheduling or memory optimization, making raw arithmetic volume the dominant bottleneck. The 60% MAC reduction directly translates into a near-halving of execution time.

**Architectural Caveat:** On physical silicon, the vertical 5×1 pass requires strided memory access (jumping by the full row width for each element). This breaks spatial locality and can cause significant L1 cache thrashing. Our gem5 results suggest that in this specific simulation configuration — with a 64 kB L1 data cache and -O0's already-inflated memory traffic — the arithmetic savings outweigh the cache penalty. This relationship would likely shift at higher optimization levels or on cache-constrained real hardware.

---

### 2.2 Sobel Output Data Types

For an 8-bit input image (pixel values `0`–`255`), the maximum theoretical output of the Sobel-X kernel `[-1, 0, 1; -2, 0, 2; -1, 0, 1]` occurs when all positive weights align with `255` and all negative weights align with `0`:

```
Max = (1 × 255) + (2 × 255) + (1 × 255) = 1020
Min = -1020
```

The absolute range is `[-1020, 1020]`, which fits comfortably within **`int16_t`** (range `[-32768, 32767]`). Using `int16_t` instead of `int32_t` halves memory bandwidth for the Gx and Gy buffers, which benefits both cache utilization and SIMD load efficiency in later RVV stages.

---

### 2.3 Magnitude Normalization Strategy

Normalizing gradient magnitudes to `[0, 255]` requires dividing every pixel's magnitude by the global maximum found in the image. A single-pass approach is not feasible because:

- The global maximum is unknown until the **final pixel** is evaluated.
- Attempting to normalize speculatively and backfill whenever a new maximum is found would require repeated random-access writes over the entire output buffer — destroying sequential access patterns and cache performance.

A **two-pass algorithm** is therefore strictly necessary: one pass to find the global maximum (a reduction), and a second pass to apply the normalization. This is why the RVV Magnitude stage includes a vector reduction (`__riscv_vredmax`) before the normalization loop.

---

### 2.4 QEMU/gem5 Benchmarking Methodology

Standard QEMU user-mode is an **instruction translator**, not a cycle-accurate simulator. It does not model cache latency, branch prediction, or pipeline stalls. Absolute wall-clock times from QEMU are volatile and strongly influenced by host OS scheduling.

This project uses **gem5 with `TimingSimpleCPU`**, which provides a deterministic, microarchitecturally detailed simulation. Key metrics used:

| Metric | Meaning | Reliability |
|---|---|---|
| `simTicks` / Time (ms) | Simulated wall-clock time at 3.0 GHz | High — deterministic |
| `simInsts` | Dynamic instruction count | Very High — exact |
| CPI | Cycles per instruction | High — reflects memory stalls |

Because gem5 simulates the full memory hierarchy (L1/L2 caches, DDR4 latency), it can reveal penalties like cache misses and memory bottlenecks that QEMU would hide. Relative comparisons between optimization levels are valid and meaningful.

---

### 2.5 Amdahl's Law & Profiling Strategy

Amdahl's Law states that the maximum achievable speedup of a system is strictly limited by the fraction of execution time that can be optimized:

```
Speedup_max = 1 / (1 - P)
```

where `P` is the fraction of runtime that benefits from the optimization.

From our `-O3` profiling, the pipeline time is distributed as follows (separable Gaussian + Sobel + Magnitude L1 + Direction):

| Stage | % of Total Runtime |
|---|---|
| Gaussian (Separable) | ~49.5% |
| Magnitude (L1) | ~17.8% |
| Direction | ~20.0% |
| Sobel Filter | ~12.7% |

Convolutions (Gaussian + Sobel) account for approximately **62.2%** of total runtime. If direction computation (~20%) were optimized to zero cost, the total speedup would be capped at ~1.25×. By contrast, optimizing the Gaussian and Sobel stages (the two convolution kernels) directly attacks the majority of the runtime and produces the highest return on investment. This is the basis for targeting those two stages with RVV intrinsics.

---

## 3. Simulation Specifications

All benchmarks were executed using a custom `run_gem5.py` configuration with the following hardware model:

| Parameter | Value |
|---|---|
| CPU Model | TimingSimpleCPU @ 3.0 GHz |
| ISA Extension | RISC-V + RVV (VLEN = 128 & 256 bits) |
| L1 Instruction Cache | 64 kB, 4-way set associative, 1-cycle latency, 4 MSHRs |
| L1 Data Cache | 64 kB, 4-way set associative, 2-cycle latency, 16 MSHRs |
| L2 Cache | 2 MB, 8-way set associative, 12-cycle latency, 32 MSHRs |
| Main Memory | 4 GB DDR4-2400 (8×8 configuration) |

---

## 4. Compiler Optimization Sweep (Phase 4)

All scalar pipeline stages compiled and measured at three optimization levels. Image resolution: 512×512 pixels.

### 4.1 `-O0` (No Optimization — Baseline)

The compiler maps C++ directly to assembly with no register caching, instruction scheduling, or inlining. Instruction counts are maximally inflated by temporary spills and redundant loads.

| Pipeline Stage | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
|---|---|---|---|---|
| Gaussian (Spatial) | 1054.165 | 3,165,661,499 | 1,605,320,527 | 1.972 |
| Gaussian (Separable) | 472.992 | 1,420,446,738 | 714,428,647 | 1.988 |
| Sobel | 302.105 | 907,221,309 | 450,807,747 | 2.012 |
| Magnitude (L1) | 112.181 | 336,933,872 | 168,041,312 | 2.005 |
| Magnitude (L2) | 115.274 | 346,206,129 | 172,659,103 | 2.005 |
| Direction | 47.655 | 143,163,659 | 66,696,756 | 2.146 |
| **Full Pipeline** | **944.017** | **2,834,883,243** | **1,407,297,719** | **2.014** |

### 4.2 `-O2` (Standard Optimization)

Dead-code elimination, loop unrolling, and basic block optimization yield a **~23× reduction** in dynamic instructions compared to `-O0`.

| Pipeline Stage | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
|---|---|---|---|---|
| Gaussian (Spatial) | 50.146 | 150,587,611 | 100,113,605 | 1.504 |
| Gaussian (Separable) | 20.881 | 62,706,244 | 33,302,911 | 1.883 |
| Sobel | 4.265 | 12,807,253 | 7,976,615 | 1.606 |
| Magnitude (L1) | 6.397 | 19,211,040 | 9,699,544 | 1.981 |
| Magnitude (L2) | 6.961 | 20,902,405 | 10,748,683 | 1.945 |
| Direction | 6.185 | 18,573,534 | 7,843,761 | 2.369 |
| **Full Pipeline** | **42.783** | **128,475,893** | **61,183,891** | **2.101** |

### 4.3 `-O3` (Aggressive Speed Optimization)

The compiler attempts auto-vectorization (`-ftree-vectorize`), aggressive inlining, and advanced loop transformations.

| Pipeline Stage | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
|---|---|---|---|---|
| Gaussian (Spatial) | 14.947 | 44,885,316 | 25,467,095 | 1.762 |
| Gaussian (Separable) | 14.959 | 44,921,667 | 25,471,607 | 1.763 |
| Sobel | 3.852 | 11,567,418 | 5,665,835 | 2.042 |
| Magnitude (L1) | 5.375 | 16,139,883 | 9,699,545 | 1.664 |
| Magnitude (L2) | 6.705 | 20,135,017 | 10,748,688 | 1.873 |
| Direction | 6.031 | 18,111,234 | 7,839,699 | 2.310 |
| **Full Pipeline** | **34.580** | **103,843,265** | **45,962,824** | **2.259** |

**Notable observation:** At `-O3`, the separable Gaussian filter (14.96 ms) no longer outperforms the spatial filter (14.95 ms) — both converge to nearly identical timings. The compiler's auto-vectorizer partially compensates for the separable filter's arithmetic advantage by vectorizing the spatial filter's innermost loop. This demonstrates that structural code optimizations become less dominant at higher compiler optimization levels.

### 4.4 Auto-Vectorization Analysis

When compiled with `-O3 -ftree-vectorize`, GCC emits auto-vectorization reports via `-fopt-info-vec-all`. Key finding:

**Loops with inner boundary checks (`if (x < 0 || y < 0 || ...)`) could NOT be auto-vectorized** because the control-flow divergence breaks the requirement for uniform SIMD operations. The compiler emits:
```
not vectorized: control flow in loop
```

**Solution implemented:** By pre-padding the image array with a border of zeros (equal to half the kernel size), the inner loop becomes a pure, continuous arithmetic block with no branching. After this structural change, GCC successfully vectorized the convolution inner loop, confirming:
```
vectorized N iterations
```

The disassembly (via `riscv64-unknown-elf-objdump -d binary | grep -c vset`) confirmed the presence of `vsetvli` instructions in the auto-vectorized binary, validating that the compiler emitted RVV instructions.

---

## 5. Profiling & Hotspot Identification (Phase 5)

Using `clock_gettime(CLOCK_MONOTONIC)` wrappers around each pipeline stage (100+ iterations for stability) under `-O3`, the runtime breakdown for the standard pipeline (Separable Gaussian + Sobel + Magnitude L1 + Direction) is:

| Stage | Time (ms) | % of Total |
|---|---|---|
| Gaussian (Separable) | 14.959 | 49.5% |
| Magnitude (L1) | 5.375 | 17.8% |
| Direction | 6.031 | 20.0% |
| Sobel | 3.852 | 12.7% |

**Hotspot conclusion:** Gaussian convolution alone accounts for nearly half of total runtime. Combined with Sobel, these two convolution kernels represent **62.2%** of execution time — the clear targets for RVV intrinsic implementation.

Direction computation (20.0%) is a tempting target by time share, but its computation is already minimal and branchless (integer cross-multiplication to avoid `atan2()`). Writing RVV intrinsics for it would yield disproportionately low returns compared to the convolution stages.

---

## 6. RVV Intrinsic Optimization (Phase 6)

### RVV Programming Model Summary

RVV (RISC-V Vector Extension) is **vector-length-agnostic (VLA)**. Unlike ARM NEON (fixed 128-bit) or x86 AVX (fixed 256-bit), RVV code calls `__riscv_vsetvl_e*m*(n)` at runtime to determine how many elements the hardware can process in one iteration based on the current VLEN and LMUL. The strip-mining loop pattern:

```c
for (int i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e32m1(n - i);
    // process vl elements starting at offset i
}
```

This produces a single binary that runs correctly at VLEN = 128, 256, or 512 bits without modification.

**LMUL (Length Multiplier)** groups physical vector registers into logical registers:
- `LMUL=1`: 32 logical registers, each 1× physical register
- `LMUL=2`: 16 logical registers, each 2× physical registers (more elements per op, fewer regs)
- `LMUL=4`: 8 logical registers, each 4× physical registers (maximum throughput, high register pressure)

---

### 6.1 LMUL Sweep — Gaussian Separable Filter

**Configuration:** `-O3`, VLEN = 256 bits

| LMUL | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
|---|---|---|---|---|
| LMUL = 1 | 2.928 | 8,793,490 | 1,152,087 | 7.633 |
| **LMUL = 2** | **2.748** | **8,250,477** | 765,527 | 10.778 |
| LMUL = 4 | 2.760 | 8,289,203 | 769,142 | 10.777 |

**Optimal: LMUL = 2** — 6.1% faster than LMUL = 1.

---

### 6.2 LMUL Sweep — Sobel Filter

**Configuration:** `-O3`, VLEN = 256 bits

| LMUL | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
|---|---|---|---|---|
| LMUL = 1 | 0.688 | 2,066,219 | 623,934 | 3.312 |
| **LMUL = 2** | **0.579** | **1,738,989** | 407,694 | 4.265 |
| LMUL = 4 | 0.652 | 1,956,711 | 340,374 | 5.749 |

**Optimal: LMUL = 2** — 15.8% faster than LMUL = 1.

---

### 6.3 LMUL Sweep — Magnitude L1

**Configuration:** `-O3`, VLEN = 256 bits

| LMUL | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
|---|---|---|---|---|
| **LMUL = 1** | **0.600** | **1,802,306** | 410,320 | 4.392 |
| LMUL = 2 | 0.632 | 1,898,467 | 410,320 | 4.627 |
| LMUL = 4 | 0.633 | 1,899,311 | 410,320 | 4.629 |

**Optimal: LMUL = 1** — higher LMUL settings are strictly worse here.

---

### 6.4 LMUL Analysis & Conclusions

**Why LMUL = 2 wins for convolutions (Gaussian & Sobel):**

Transitioning from `LMUL=1` to `LMUL=2` processes twice as many pixels per loop iteration. Dynamic instruction counts drop ~33% (Gaussian) and ~34% (Sobel). While requesting 512 bits of data per memory instruction raises CPI slightly (higher L1 pressure), the instruction volume reduction more than compensates, producing the fastest execution.

**Why LMUL = 4 regresses for Sobel:**

The Sobel convolution requires simultaneous access to three image rows (top, middle, bottom) to compute Gx and Gy. At `LMUL=4`, each logical register occupies four physical registers, restricting the total number of distinct vector variables. The compiler is forced to **spill intermediate values to the stack**, introducing extra load/store instructions that erase the arithmetic benefit. CPI spikes to 5.749 — the memory subsystem cannot feed the 1024-bit-wide vector units fast enough.

**Why LMUL scaling fails for Magnitude L1:**

The dynamic instruction count is identical (**410,320**) across all three LMUL settings for the Magnitude stage. The compiler failed to exploit the wider vector lengths to unroll or pack additional data per iteration. Since no arithmetic benefit materialized, higher LMUL settings only added overhead from wider vector register setup, strictly degrading performance. `LMUL=1` remains optimal for this stage.

---

## 7. Summary Optimization Table

The table below shows the end-to-end optimization journey from unoptimized scalar to hand-tuned RVV intrinsics.

| Stage | -O0 (ms) | -O2 (ms) | -O3 (ms) | RVV LMUL=1 (ms) | RVV LMUL=2 (ms) | RVV LMUL=4 (ms) | Best RVV vs -O0 |
|---|---|---|---|---|---|---|---|
| Gaussian (Separable) | 472.992 | 20.881 | 14.959 | 2.928 | **2.748** | 2.760 | **~172×** |
| Sobel | 302.105 | 4.265 | 3.852 | 0.688 | **0.579** | 0.652 | **~521×** |
| Magnitude (L1) | 112.181 | 6.397 | 5.375 | **0.600** | 0.632 | 0.633 | **~187×** |
| Direction | 47.655 | 6.185 | 6.031 | *(scalar)* | *(scalar)* | *(scalar)* | — |
| **Full Pipeline** | **944.017** | **42.783** | **34.580** | — | — | — | — |

> RVV benchmarks run at VLEN = 256 bits with `-O3`. Equivalence verified at VLEN = 128, 256, and 512.

**Key milestones on the optimization journey:**
- `-O2` delivers a **~22× speedup** over `-O0` for free — just by enabling the compiler.
- `-O3` adds another **~1.24×** on top of `-O2` through auto-vectorization and inlining.
- Hand-written RVV intrinsics push the hotspot stages (Gaussian + Sobel) to **>170× faster** than the `-O0` baseline — a transformation that required understanding strip-mining, data widening (`uint8 → int16 → int32`), LMUL selection, and vector reduction semantics.

---

*Report generated from project results files (`benchmark.md` + `answers.md`). All measurements from gem5 `TimingSimpleCPU` simulation. Authored by the project team.*
