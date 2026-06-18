# RISC-V Vector (RVV) Canny Edge Detection - Benchmarks

## 1. Simulation Specifications (`gem5`)

Unlike standard QEMU user-mode emulators, this project utilizes **gem5** to provide a deterministic, microarchitecturally detailed simulation. The benchmarks were executed using a custom `run_gem5.py` configuration with the following hardware specifications:

* **CPU Model:** `TimingSimpleCPU` @ 3.0 GHz
* **ISA Extension:** RISC-V with RVV enabled (**VLEN = 128 & 256 bits**)
* **L1 Instruction Cache:** 64 kB, 4-way set associative, 1-cycle latency (4 MSHRs)
* **L1 Data Cache:** 64 kB, 4-way set associative, 2-cycle latency (16 MSHRs)
* **L2 Cache:** 2 MB, 8-way set associative, 12-cycle latency (32 MSHRs)
* **Main Memory:** 4 GB DDR4-2400 (8x8)

---

## 2. Measurement Methodology: gem5 vs. QEMU

The project brief notes that standard QEMU user-mode is not cycle-accurate and relies on volatile wall-clock timing (`clock_gettime`). However, because we are using **gem5's `TimingSimpleCPU` and a defined memory hierarchy**, our measurements are highly deterministic.

We do not need to rely on host OS wall-clock time. Instead, we rely on gem5's `simTicks` (simulated cycles) and `simInsts` (simulated dynamic instruction counts). This allows us to precisely observe the penalties of cache misses and memory latency that a simple instruction translator (like QEMU) would ignore.

---

## 3. Compiler Optimization Sweep (Scalar Baseline)

The following tables document the C++ scalar pipeline performance across three compiler optimization tiers.

### Phase 4 Data: `-O0` (No Optimization - Baseline)

*The compiler maps C++ directly to assembly with no register caching, instruction scheduling, or inlining. Instruction counts are massively inflated.*

| Pipeline Stage | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
| --- | --- | --- | --- | --- |
| Stage 0: Gaussian (Spatial) | 1054.165 | 3,165,661,499 | 1,605,320,527 | 1.972 |
| Stage 1: Gaussian (Separable) | 472.992 | 1,420,446,738 | 714,428,647 | 1.988 |
| Stage 2: Sobel | 302.105 | 907,221,309 | 450,807,747 | 2.012 |
| Stage 3: Magnitude (L1) | 112.181 | 336,933,872 | 168,041,312 | 2.005 |
| Stage 4: Magnitude (L2) | 115.274 | 346,206,129 | 172,659,103 | 2.005 |
| Stage 5: Direction | 47.655 | 143,163,659 | 66,696,756 | 2.146 |
| **Full Pipeline** | **944.017** | **2,834,883,243** | **1,407,297,719** | **2.014** |

### Phase 4 Data: `-O2` (Standard Optimization)

*The compiler performs dead-code elimination, loop unrolling, and basic block optimization, resulting in a ~23x reduction in instructions compared to `-O0`.*

| Pipeline Stage | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
| --- | --- | --- | --- | --- |
| Stage 0: Gaussian (Spatial) | 50.146 | 150,587,611 | 100,113,605 | 1.504 |
| Stage 1: Gaussian (Separable) | 20.881 | 62,706,244 | 33,302,911 | 1.883 |
| Stage 2: Sobel | 4.265 | 12,807,253 | 7,976,615 | 1.606 |
| Stage 3: Magnitude (L1) | 6.397 | 19,211,040 | 9,699,544 | 1.981 |
| Stage 4: Magnitude (L2) | 6.961 | 20,902,405 | 10,748,683 | 1.945 |
| Stage 5: Direction | 6.185 | 18,573,534 | 7,843,761 | 2.369 |
| **Full Pipeline** | **42.783** | **128,475,893** | **61,183,891** | **2.101** |

### Phase 4 Data: `-O3` (Aggressive Speed Optimization)

*The compiler attempts auto-vectorization (`-ftree-vectorize`) and aggressive function inlining.*

| Pipeline Stage | Time (ms) | Cycle Count | Instructions (simInsts) | CPI |
| --- | --- | --- | --- | --- |
| Stage 0: Gaussian (Spatial) | 14.947 | 44,885,316 | 25,467,095 | 1.762 |
| Stage 1: Gaussian (Separable) | 14.959 | 44,921,667 | 25,471,607 | 1.763 |
| Stage 2: Sobel | 3.852 | 11,567,418 | 5,665,835 | 2.042 |
| Stage 3: Magnitude (L1) | 5.375 | 16,139,883 | 9,699,545 | 1.664 |
| Stage 4: Magnitude (L2) | 6.705 | 20,135,017 | 10,748,688 | 1.873 |
| Stage 5: Direction | 6.031 | 18,111,234 | 7,839,699 | 2.310 |
| **Full Pipeline** | **34.580** | **103,843,265** | **45,962,824** | **2.259** |

---

## 4. Architectural Analysis & Auto-Vectorization (Phase 4)

### Separable vs. Spatial Filters in gem5

Decomposing a 5x5 Gaussian into a separable 1x5 and 5x1 pass reduces the multiply-accumulate operations from 25 to 10 per pixel. While the vertical 5x1 pass causes strided memory access that can hurt physical cache performance, our `gem5` simulation at `-O0` shows the separable filter executing in less than half the time (472 ms vs 1054 ms). Because `-O0` forces so many redundant memory loads, the 60% reduction in raw mathematical operations heavily outweighs the penalty of L1 cache misses in this specific `gem5` configuration.

### Auto-Vectorization & Boundary Checks

When compiling with `-O3 -ftree-vectorize`, GCC attempts to map scalar loops to RVV SIMD instructions. However, boundary condition checks (e.g., `if (x < 0 || y < 0)`) placed inside the innermost convolution loop introduce control-flow divergence. SIMD hardware requires uniform operations across all data lanes; therefore, inner-loop branching forces the compiler to emit a "not vectorized: control flow in loop" warning.

By removing the boundary checks and instead pre-padding the image array with a border of zeros, the inner loop becomes a pure, continuous block of math. This structural code change is necessary to allow the compiler to successfully auto-vectorize the algorithm.

---

## 5. Profiling & Amdahl's Law (Phase 5)

Using our fastest scalar baseline (`-O3`), we can profile the percentage of total execution time consumed by each stage of the standard separable pipeline (Gaussian Separable + Sobel + Magnitude L1 + Direction):

* **Gaussian (Separable):** ~49.5%
* **Sobel Filter:** ~12.7%
* **Magnitude (L1):** ~17.8%
* **Direction:** ~20.0%

### Applying Amdahl's Law

Amdahl's Law dictates that the theoretical maximum speedup of a system is strictly limited by the fraction of the code that cannot be optimized. According to the profiling data, convolutions (Gaussian and Sobel) account for **~62.2%** of the execution time.

If we were to spend hours writing highly optimized RVV intrinsics for a minor stage, the global speedup would be mathematically capped. By focusing our intrinsic optimization efforts on the Gaussian and Sobel stages, we target the mathematical bottleneck of the pipeline, ensuring our manual vectorization yields the highest possible return on investment.

