# Canny Edge Detection on RISC-V using RVV
**Target Architecture:** `rv64gcv` (QEMU User-mode)

## Overview
This project implements a high-performance Canny edge detection pipeline, focusing on the migration from a scalar C++ baseline to a hand-optimized implementation using RISC-V Vector (RVV) intrinsics.

### Pipeline Stages
1. **Gaussian Blur:** 5x5 Kernel (2D Convolution)
2. **Sobel Operator:** $G_x$ and $G_y$ gradients (3x3 Kernels)
3. **Magnitude & Direction:** $L_1/L_2$ Norms and 4-way direction quantization.
4. **NMS & Hysteresis:** Non-Maximum Suppression and Double Thresholding.

### Project Workflow
The development process follows a standard embedded systems optimization lifecycle:
* **Baseline:** Develop a clean, portable scalar C++ implementation for functional reference.
* **Measurement:** Profile execution and analyze compiler output (from `-O0` to `-Ofast`) to identify bottlenecks.
* **Optimization:** Rewrite performance-critical kernels using RVV intrinsics.
* **Verification:** Ensure bit-exact equivalence between scalar and vector outputs across various Vector Lengths (VLEN).

---

## Technical Stack
* **Environment:** WSL2 (Ubuntu 24.04) or Arch Linux.
* **Toolchain:** `riscv64-linux-gnu-g++` (configured with `--with-arch=rv64gcv`).
* **Emulation:** QEMU with RVV 1.0 support.
* **Testing:** GoogleTest for host-side logic and assert-based testing for target emulation.
* **Documentation:** Doxygen.

---

## Hardware-Software Interface
While the project runs in emulation, the RVV optimizations interact with simulated digital hardware structures:

* **Vector Datapath (VPUs):** Intrinsics like `vadd` utilize multiple ALUs in parallel, processing data batches rather than single pixels.
* **Vector Register File (VRF):** Testing across different `VLEN` values simulates various hardware tiers.
* **Length Multiplier (LMUL):** Grouping registers allows the hardware to handle larger continuous data chunks.
* **Vector Length Agnostic (VLA) Programming:** Utilizing `vsetvli` interacts with the `vl` Control and Status Register (CSR), allowing the logic to adapt to the physical hardware width dynamically.

---

## Project Structure
```plaintext
.
├── assets/         # Input/Output raw images (width*height bytes)
├── build/          # Build artifacts (host/ for x86, target/ for RISC-V)
├── include/        # Header files (.hpp)
├── src/            # Implementation files (.cpp)
├── scripts/        # Automated scripts
├── tests/          # GoogleTest (host) and RVV equivalence tests
├── tools/          # Custom profiling or visualization tools
├── Makefile        # Dual-target build system
└── Doxyfile        # Doxygen configuration
```

---

## Getting Started

### Environment Setup
The setup script automates the installation of the RISC-V GNU toolchain, QEMU, and GoogleTest. Windows OS users should use WSL2 with Ubuntu 24.04.

Run in terminal:
```bash
chmod +x scripts/phase_one_setup.sh
./scripts/phase_one_setup.sh
source ~/.bashrc
```

### Build System
The Makefile supports cross-compilation and automated testing.

| Command | Action | Purpose |
| :--- | :--- | :--- |
| `make all` | Full Build | Compiles the main pipeline (RISC-V) and runs Host tests. |
| `make canny_rv` | Target Build | Compiles the main Canny pipeline into a RISC-V `.elf`. |
| `make test` | Host Test | Compiles and runs GoogleTest natively for logic verification. |
| `make run` | Emulate | Executes the main pipeline on QEMU with `vlen=512`. |
| `make run-test NAME=x` | Unit Emulate | Compiles and runs `tests/x.cpp` on QEMU (e.g., `make run-test NAME=sobel`). |
| `make list-tests` | Discovery | Scans the `tests/` folder and lists all testable `.cpp` files. |
| `make docs` | Documentation | Generates browsable HTML API documentation via Doxygen. |
| `make clean` | Cleanup | Wipes the `build/` directory and resets the environment. |

**Configuration Overrides:**
You can override hardware parameters directly from the command line:
```bash
make run VLEN=128  # Simulate lower-end hardware
make run VLEN=256  # Simulate mid-range hardware
```

---

## Authors
* A. Atta
* Ahmed Ameer
* Aly Gano
* Marawan Mohamed
* Mouaz Kfafy
* Mohamed elHosary

## Resources & References
* [RISC-V Vector Quick Intro](https://blog.timhutt.co.uk/riscv-vector/#the-end)
* [RISCV-V RVV Intrinsics Viewer](https://dzaima.github.io/intrinsics-viewer/)
* [Doxygen Documentation](https://www.doxygen.nl/)
* [Markdown Guide](https://www.markdownguide.org/cheat-sheet/)
* [Cornell University: RISC-V Vector Extension](https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/rvv-llvm-gisel/)
* [Gaussian Blur](https://computergraphics.stackexchange.com/questions/39/how-is-gaussian-blur-implemented)
