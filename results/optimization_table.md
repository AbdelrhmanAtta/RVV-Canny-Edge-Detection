## Phase 4: Compiler Optimization Sweep

### 4.1 Execution Time Analysis

To evaluate the impact of compiler optimizations on the scalar Canny edge detection pipeline, the code was compiled across five standard GCC optimization levels (`-O0`, `-O2`, `-O3`, `-Os`, `-Ofast`). Benchmarks were executed on QEMU (RV64GCV, VLEN=512) over 100 iterations using a 512x512 raw image to ensure stable wall-clock timing.

**Table 1: Execution Time (ms/iteration)**

| Pipeline Stage | `-O0` (Unoptimized) | `-O2` | `-O3` (Speed) | `-Os` (Size) | `-Ofast` |
| --- | --- | --- | --- | --- | --- |
| **Gaussian Spatial** | 1342.077 | 17.135 | 7.372 | 19.518 | 7.478 |
| **Gaussian Separable** | 702.679 | 7.062 | 3.977 | 8.795 | 4.036 |
| **Sobel (3x3)** | 5.755 | 2.475 | 2.442 | 1.969 | 2.546 |
| **Magnitude L1** | 161.299 | 7.072 | 6.733 | 7.033 | 6.788 |
| **Magnitude L2** | 197.638 | 11.853 | 12.203 | 34.805 | 15.351 |
| **Direction** | 59.017 | 3.067 | 3.084 | 3.097 | 3.133 |

**Key Observations:**

* **The Baseline Imperative:** The unoptimized `-O0` build of the Gaussian spatial filter took 1342 ms, while `-O3` reduced this to 7.37 ms—an extraordinary **~182x speedup**. This confirms that any hardware acceleration claims (like RISC-V Vector extensions) must be measured against an aggressively optimized scalar baseline (`-O3`), otherwise the speedups reported would be artificially inflated.
* **Mathematical Complexity:** The L2 Magnitude computation is consistently slower than L1 (e.g., 12.2 ms vs. 6.7 ms at `-O3`). The L2 norm requires costly floating-point conversions and a square root operation (`std::sqrt`), whereas L1 relies purely on integer absolute addition. Notably, at `-Os`, the L2 time spikes to 34.8 ms because the compiler refuses to inline the `sqrt` function to save binary size, resulting in heavy function-call overhead inside the inner loop.
* **Floating Point Relaxation:** The `-Ofast` flag (which enables `-ffast-math`) yielded negligible differences compared to `-O3`. Because the core pipeline is built on integer arithmetic, relaxing IEEE-754 floating-point compliance provided no tangible benefit.

### 4.2 Binary Size and Dead Code Elimination

The pipeline was compiled with the `-static` flag, meaning the C++ standard library (`libstdc++`) is bundled directly into the executable.

**Table 2: Binary Footprint**

| Flag | Binary Size |
| --- | --- |
| `-O0` | 3.6 MB |
| `-Os` | 3.0 MB |
| `-O2`, `-O3`, `-Ofast` | 2.3 MB |

**Analysis of the Size Anomaly:**
Typically, `-O3` produces larger binaries than `-Os` due to loop unrolling and function inlining. However, our results show `-O3` is 0.7 MB *smaller* than `-Os`. Because we are statically linking the C++ standard library, the unoptimized and `-Os` builds retain significant amounts of unused template instantiation boilerplate. At `-O3`, GCC performs much more aggressive **Dead Code Elimination (DCE)**, successfully stripping out the unused library bloat and resulting in a leaner binary.

### 4.3 The Failure of Auto-Vectorization

An analysis of the GCC vectorization logs (`-fopt-info-vec-all` at `-O3`) revealed that the compiler failed to automatically vectorize the core image processing loops. The failures were primarily caused by:

1. **Complex Loop Nests:** GCC cannot automatically unroll and flatten the 4-deep 2D windowing loops required for convolution (Gaussian and Sobel).
2. **Control Flow:** The branching `if/else` statements in the Direction stage prevented Safe Single Instruction Multiple Data (SIMD) mapping.

Consequently, while the compiler optimized the scalar instructions brilliantly, it left the RISC-V Vector unit entirely unutilized. Achieving true hardware acceleration will require manual implementation using RVV C++ intrinsics.

---

## Phase 5: Profiling and Amdahl's Law

To determine the most effective strategy for the RVV vectorization phase, the `-O3` execution times of the optimal, integer-only pipeline (Separable Gaussian, Sobel, L1 Magnitude, Direction) were profiled.

**Optimal Pipeline Total Execution Time:** 16.236 ms/iteration

| Pipeline Stage | Time (ms/iter) | % of Total Time | Optimization Priority |
| --- | --- | --- | --- |
| **Magnitude L1** | 6.733 ms | 41.5% | **High** |
| **Gaussian (Separable)** | 3.977 ms | 24.5% | **High** |
| **Direction** | 3.084 ms | 19.0% | Low |
| **Sobel (3x3)** | 2.442 ms | 15.0% | Low |

### 5.1 Algorithmic Refinement

Before vectorizing, the algorithm itself was optimized. The 2D Spatial Gaussian (7.372 ms) was replaced with a 1D Separable Gaussian (3.977 ms). By mathematically decomposing the 5x5 kernel into two passes, the execution time was nearly cut in half before a single vector instruction was written. **The Separable Gaussian will serve as the baseline for the RVV implementation.**

### 5.2 Application of Amdahl's Law

Amdahl's Law dictates that the theoretical speedup of a system is strictly limited by the unoptimized fraction of the task.

According to our profiling data, **Magnitude L1** and the **Separable Gaussian** blur account for **66%** of the pipeline's total execution time. These two stages represent the primary bottlenecks.

Conversely, the Sobel gradient and Direction stages account for only 34% of the time combined. Even if manual RVV vectorization made the Sobel filter execute instantaneously (a 0 ms execution time), the maximum theoretical speedup for the entire pipeline would only be roughly 1.17x.

**Conclusion for Phase 6:** Development efforts for hardware vectorization will be strictly focused on the **Magnitude L1** and **Separable Gaussian** implementations. Writing complex RVV intrinsics for the Sobel or Direction stages would yield a poor return on engineering time.