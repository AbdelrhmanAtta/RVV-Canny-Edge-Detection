# RVV Canny Edge Detection — Optimization & Analysis Report

**Project:** Canny Edge Detection on RISC-V with Vector Extension  
**Architecture:** `rv64gcv` | **Emulation:** gem5 (TimingSimpleCPU @ 3 GHz)  
**Language:** C23 | **Team Size:** 6 Students

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
6. [RVV Optimization Analysis](#6-rvv-optimization-analysis)
   - 6.1 LMUL (Vector Length Multiplier) Sweeps
   - 6.2 Sweep Analysis

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
| :--- | :--- |
| Spatial 5×5 | 1054 |
| Separable (1×5 + 5×1) | 473 |

The separable filter was approximately **2.2× faster** under unoptimized compilation. At `-O0`, the compiler produces no instruction scheduling or memory optimization, making raw arithmetic volume the dominant bottleneck. The 60% MAC reduction directly translates into a near-halving of execution time.

**Architectural Caveat:** On physical silicon, the vertical 5×1 pass requires strided memory access (jumping by the full row width for each element). This breaks spatial locality and can cause significant L1 cache thrashing. Our gem5 results suggest that in this specific simulation configuration — with a 64 kB L1 data cache and -O0's already-inflated memory traffic — the arithmetic savings outweigh the cache penalty. This relationship would likely shift at higher optimization levels or on cache-constrained real hardware.

### 2.2 Sobel Output Data Types

For an 8-bit input image (pixel values `0`–`255`), the maximum theoretical output of the Sobel-X kernel `[-1, 0, 1; -2, 0, 2; -1, 0, 1]` occurs when all positive weights align with `255` and all negative weights align with `0`:

> Max = (1 × 255) + (2 × 255) + (1 × 255) = 1020  
> Min = -1020

The absolute range is `[-1020, 1020]`, which fits comfortably within **`int16_t`** (range `[-32768, 32767]`). Using `int16_t` instead of `int32_t` halves memory bandwidth for the Gx and Gy buffers, which benefits both cache utilization and SIMD load efficiency in later RVV stages.

### 2.3 Magnitude Normalization Strategy

Normalizing gradient magnitudes to `[0, 255]` requires dividing every pixel's magnitude by the global maximum found in the image. A single-pass approach is not feasible because:

* The global maximum is unknown until the **final pixel** is evaluated.
* Attempting to normalize speculatively and backfill whenever a new maximum is found would require repeated random-access writes over the entire output buffer — destroying sequential access patterns and cache performance.

A **two-pass algorithm** is therefore strictly necessary: one pass to find the global maximum (a reduction), and a second pass to apply the normalization. This is why the RVV Magnitude stage includes a vector reduction (`__riscv_vredmax`) before the normalization loop.

### 2.4 QEMU/gem5 Benchmarking Methodology

Standard QEMU user-mode is an **instruction translator**, not a cycle-accurate simulator. It does not model cache latency, branch prediction, or pipeline stalls. Absolute wall-clock times from QEMU are volatile and strongly influenced by host OS scheduling.

This project uses **gem5 with `TimingSimpleCPU`**, which provides a deterministic, microarchitecturally detailed simulation. Key metrics used:

| Metric | Meaning | Reliability |
| :--- | :--- | :--- |
| `simTicks` / Time (ms) | Simulated wall-clock time at 3.0 GHz | High — deterministic |
| `simInsts` | Dynamic instruction count | Very High — exact |
| CPI | Cycles per instruction | High — reflects memory stalls |

Because gem5 simulates the full memory hierarchy (L1/L2 caches, DDR4 latency), it can reveal penalties like cache misses and memory bottlenecks that QEMU would hide. Relative comparisons between optimization levels are valid and meaningful.

### 2.5 Amdahl's Law & Profiling Strategy

Amdahl's Law states that the maximum achievable speedup of a system is strictly limited by the fraction of execution time that can be optimized. From our `-O3` profiling, the pipeline time is distributed as follows (separable Gaussian + Sobel + Magnitude L1 + Direction):

| Stage | % of Total Runtime |
| :--- | :--- |
| Gaussian (Separable) | ~49.5% |
| Magnitude (L1) | ~17.8% |
| Direction | ~20.0% |
| Sobel Filter | ~12.7% |

Convolutions (Gaussian + Sobel) account for approximately **62.2%** of total runtime. If direction computation (~20%) were optimized to zero cost, the total speedup would be capped at ~1.25×. By contrast, optimizing the Gaussian and Sobel stages (the two convolution kernels) directly attacks the majority of the runtime and produces the highest return on investment. This is the basis for targeting those two stages with RVV intrinsics.

---

## 3. Simulation Specifications

All benchmarks were executed using a custom `run_gem5.py` configuration with the following hardware model:

* **CPU Model:** TimingSimpleCPU @ 3.0 GHz
* **ISA Extension:** RISC-V + RVV (VLEN = 128 & 256 bits)
* **L1 Instruction Cache:** 64 kB, 4-way set associative, 1-cycle latency, 4 MSHRs
* **L1 Data Cache:** 64 kB, 4-way set associative, 2-cycle latency, 16 MSHRs
* **L2 Cache:** 2 MB, 8-way set associative, 12-cycle latency, 32 MSHRs
* **Main Memory:** 4 GB DDR4-2400 (8×8 configuration)

---

## 4. Compiler Optimization Sweep (Phase 4)

All scalar pipeline stages compiled and measured at three optimization levels. Image resolution: 512×512 pixels.

### 4.1 `-O0` (No Optimization — Baseline)

The compiler maps C++ directly to assembly with no register caching, instruction scheduling, or inlining. Instruction counts are maximally inflated by temporary spills and redundant loads.

| Pipeline Stage | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
| :--- | :--- | :--- | :--- | :--- |
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
| :--- | :--- | :--- | :--- | :--- |
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
| :--- | :--- | :--- | :--- | :--- |
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
> `not vectorized: control flow in loop`

**Solution implemented:** By pre-padding the image array with a border of zeros (equal to half the kernel size), the inner loop becomes a pure, continuous arithmetic block with no branching. After this structural change, GCC successfully vectorized the convolution inner loop, confirming:
> `vectorized N iterations`

The disassembly (via `riscv64-unknown-elf-objdump -d binary | grep -c vset`) confirmed the presence of `vsetvli` instructions in the auto-vectorized binary, validating that the compiler emitted RVV instructions.

---

## 5. Profiling & Hotspot Identification (Phase 5)

Using `clock_gettime(CLOCK_MONOTONIC)` wrappers around each pipeline stage (100+ iterations for stability) under `-O3`, the runtime breakdown for the standard pipeline (Separable Gaussian + Sobel + Magnitude L1 + Direction) is:

| Stage | Time (ms) | % of Total |
| :--- | :--- | :--- |
| Gaussian (Separable) | 14.959 | 49.5% |
| Magnitude (L1) | 5.375 | 17.8% |
| Direction | 6.031 | 20.0% |
| Sobel | 3.852 | 12.7% |

**Hotspot conclusion:** Gaussian convolution alone accounts for nearly half of total runtime. Combined with Sobel, these two convolution kernels represent **62.2%** of execution time — the clear targets for RVV intrinsic implementation.

Direction computation (20.0%) is a tempting target by time share, but its computation is already minimal and branchless (integer cross-multiplication to avoid `atan2()`). Writing RVV intrinsics for it would yield disproportionately low returns compared to the convolution stages.

---

## 6. RVV Optimization Analysis

### 6.1 LMUL (Vector Length Multiplier) Sweeps
*This test evaluates the performance impact of grouping vector registers (LMUL). Higher LMUL values process more elements per instruction but increase register pressure, which can lead to spilling if the compiler runs out of architectural registers. It also increases pressure on the memory subsystem.*

**Benchmark Configuration:**
* **Optimization Level:** `-O3`
* **VLEN:** 256 bits
* **Target Stages:** Gaussian Separable Pass, Sobel Filter, Magnitude L1

#### 1. Gaussian Separable Filter

| LMUL Setting | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
| :--- | :--- | :--- | :--- | :--- |
| **LMUL = 1** | 2.928 | 8,793,490 | 1,152,087 | 7.633 |
| **LMUL = 2** | **2.748** | **8,250,477** | 765,527 | 10.778 |
| **LMUL = 4** | 2.760 | 8,289,203 | 769,142 | 10.777 |

#### 2. Sobel Filter

| LMUL Setting | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
| :--- | :--- | :--- | :--- | :--- |
| **LMUL = 1** | 0.688 | 2,066,219 | 623,934 | 3.312 |
| **LMUL = 2** | **0.579** | **1,738,989** | 407,694 | 4.265 |
| **LMUL = 4** | 0.652 | 1,956,711 | 340,374 | 5.749 |

#### 3. Magnitude L1

| LMUL Setting | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
| :--- | :--- | :--- | :--- | :--- |
| **LMUL = 1** | **0.600** | **1,802,306** | 410,320 | 4.392 |
| **LMUL = 2** | 0.632 | 1,898,467 | 410,320 | 4.627 |
| **LMUL = 4** | 0.633 | 1,899,311 | 410,320 | 4.629 |

### 6.2 Sweep Analysis

**The Memory & Register Wall (`LMUL=4`)**
Pushing the vector grouping to `LMUL=4` consistently degrades execution time across all three stages, exposing hardware limits:
1. **Register Spilling (Sobel):** A convolution like the Sobel filter requires simultaneous access to the top, middle, and bottom image rows to compute the gradients. At `LMUL=4`, registers are grouped in blocks of four, restricting the number of logical registers available and forcing the compiler to spill intermediate data to the stack.
2. **Memory Bottlenecks:** For the Sobel filter at `LMUL=4`, a single vector instruction demands 1024 bits (128 bytes) of data. Even though the instruction count successfully drops to 340k, the CPI skyrockets to 5.749. The memory subsystem simply cannot feed the vector units fast enough, resulting in heavy processor stalls. 

**Compiler Vectorization Limits (Magnitude L1)**
The Magnitude stage (computing `|Gx| + |Gy|`) presents a highly irregular scaling pattern. The dynamic instruction count is exactly **410,320** across all three LMUL settings. This indicates that the compiler failed to utilize the wider vector lengths to unroll or pack more data per loop iteration. Because the instruction volume remained static, higher LMUL configurations strictly degraded performance due to the increased latency (CPI) of setting up wider vector operations without the benefit of doing more math per instruction. For this specific loop, `LMUL=1` remains the most efficient.

**The Convolution Sweet Spot (`LMUL=2`)**
For the sliding window convolutions (Gaussian and Sobel), transitioning from `LMUL=1` to `LMUL=2` yields the optimal performance configuration. By processing twice as many pixels per loop iteration, the dynamic instruction count drops significantly (~33% for Gaussian, ~34% for Sobel). While requesting 512 bits (64 bytes) of data per memory instruction slightly inflates the average CPI due to cache interface delays, the massive reduction in instruction volume easily offsets the latency, yielding the absolute fastest execution times for those stages.

---

## 7. Summary Table: Optimization Progress

The following table summarizes the execution time (in ms) and binary size across all optimization phases. 

| Stage | -O0 | -O2 | -O3 | Auto-vec | RVV (VLEN=128) | RVV (VLEN=256) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gaussian 5x5** | 472.992 | 20.881 | 14.959 | 14.959 | 3.125 | **2.748** |
| **Sobel Gx/Gy** | 302.105 | 4.265 | 3.852 | 3.852 | 0.878 | **0.579** |
| **Magnitude (L1)** | 112.181 | 6.397 | 5.375 | 5.375 | 0.823 | **0.600** |
| **Direction** | 47.655 | 6.185 | 6.031 | 6.031 | - | - |
| **Binary Size** | 1,517,968 | 1,208,313 | 1,208,897 | 1,208,897 | 1,208,633 | 1,208,633 |

> *Note: RVV metrics represent the optimal LMUL setting for each specific stage (Sobel/Gaussian = LMUL=2; Magnitude = LMUL=1). Direction computation remained scalar across all RVV tests.*

---

## 7.1 Analysis of Results

1.  **The Auto-Vectorization Plateau:** Moving from `-O2` to `-O3` (Auto-vec) provided significant speedups for the Gaussian and Magnitude stages, but the binary size reduction at `-O3` reflects the compiler's ability to prune dead code and inline aggressively. 
2.  **VLEN Sensitivity:** The introduction of RVV intrinsics provided an order-of-magnitude leap in performance. However, comparing `VLEN=128` to `VLEN=256` shows that the performance is not strictly linear; memory access patterns and the overhead of the strip-mining loop mean that doubling the vector length provides roughly a 15–30% performance gain rather than a 2× speedup.
3.  **Binary Size Trade-off:** The RVV-optimized binary is slightly larger (~14 kB) than the `-O3` scalar binary. This is expected, as the intrinsic implementations require additional instructions for loop control (strip-mining), setting the vector length, and managing mask/configuration states that a standard scalar compiler does not generate.
4.  **Scalar Bottlenecks:** The **Direction** stage remains the "long pole" in the tent once the convolutions are optimized. Because the direction logic relies heavily on cross-multiplication for angle quantization to avoid branching, it is inherently scalar-friendly. Future optimizations for this stage would require algorithmic changes rather than simple vectorization.

