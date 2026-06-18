### 1. Separable Filters Performance (Phase 2.2)

Decomposing a 5x5 Gaussian into separable 1x5 and 5x1 passes reduces multiply-accumulate operations from 25 to 10 per pixel. On QEMU (a user-mode emulator), this drastically reduces the dynamic instruction count, making it appear much faster. On real hardware, however, the vertical 5x1 pass requires strided memory accesses (jumping by the image's row width for every element). This breaks spatial locality and causes frequent CPU cache misses. The memory access penalty on silicon often offsets the mathematical speedup unless optimizations like loop tiling or transposition are used.
But for GEM5
A standard 5x5 spatial convolution requires 25 multiply-accumulate (MAC) operations per pixel. Decomposing this into a separable filter (a 1x5 horizontal pass followed by a 5x1 vertical pass) reduces the math to just 10 operations per pixel.

* **Observed Performance (gem5):** In our `-O0` baseline testing within gem5, the separable filter significantly outperformed the spatial filter (472 ms vs. 1054 ms). This is primarily driven by the sheer reduction in dynamic instruction count. At `-O0`, the compiler performs no instruction scheduling or memory optimization, making the raw math volume the primary bottleneck.
* **Architectural Context:** While separable filters reduce math, the vertical 5x1 pass requires strided memory access (jumping rows), which breaks spatial locality and can cause severe cache thrashing on physical hardware. Our gem5 results suggest that in this specific simulation configuration, the penalty for these potential cache misses did not outweigh the massive savings in arithmetic instruction volume.

### 2. Sobel Data Types (Phase 2.3)

For an 8-bit input image (values `0` to `255`), the maximum theoretical output of a Sobel kernel (e.g., `[-1, 0, 1]` and `[-2, 0, 2]`) occurs when the positive weights align with `255` and the negative weights align with `0`. The maximum is `(1 * 255) + (2 * 255) + (1 * 255) = 1020`, and the minimum is `-1020`. Because the absolute range is `[-1020, 1020]`, an `int16_t` (which safely covers `[-32768, 32767]`) completely prevents overflow without wasting the memory bandwidth required by `int32_t`.

### 3. Magnitude Normalization (Phase 2.4)

Normalizing gradient magnitudes to a `[0, 255]` range requires dividing every pixel's magnitude by the absolute global maximum found in the image. A single-pass approach is not straightforward because the global maximum remains unknown until the final pixel is evaluated. Attempting to normalize in a single pass would require continuously backtracking to recalculate and overwrite all previously processed pixels whenever a new maximum is discovered, which severely degrades memory performance. A two-pass algorithm is strictly necessary.

### 4. QEMU Benchmarking Limitations (Phase 4)

QEMU user-mode is an instruction translator, not a cycle-accurate hardware simulator. It does not accurately model physical microarchitecture features like cache hierarchy latency, branch prediction penalties, or pipeline stalls. As a result, absolute wall-clock timing (`ms`) is highly volatile and influenced by host OS overhead. However, dynamic instruction counts (`simInsts`) remain a highly reliable metric for relative comparisons, as they deterministically measure the exact amount of computational work the compiler generated for the architecture.

### 5. Amdahl's Law in Profiling (Phase 5)

Profiling reveals that the Gaussian blur and Sobel filter convolutions account for the vast majority of execution time. Amdahl's Law dictates that the maximum theoretical speedup of a system is limited by the fraction of code that cannot be optimized. If the gradient direction computation only consumes 8% of the runtime, infinitely optimizing it to 0.0ms will only yield an 8% overall speedup. Consequently, RVV intrinsic optimizations must be exclusively targeted at the spatial/separable convolutions to achieve meaningful performance gains.