# Canny Edge Detection on RISC-V using RVV
**Embedded Systems Project Target Architecture:** `rv64gcv` (QEMU User-mode)

## Overview
This project implements a high-performance **Canny edge detection** pipeline. The core objective is the migration from a clean C++ scalar baseline to a hand-optimized **RVV (RISC-V Vector)** intrinsic implementation.

### Pipeline Stages
1. **Gaussian Blur:** 5x5 Kernel (2D Convolution)
2. **Sobel Operator:** $G_x$ and $G_y$ gradients (3x3 Kernels)
3. **Magnitude & Direction:** $L_1/L_2$ Norms and 4-way direction quantization.
4. **NMS & Hysteresis:** (Bonus) Non-Maximum Suppression and Double Thresholding.

### The "Big Picture" Workflow
This project simulates a real embedded engineering workflow. The focus is an **optimization journey**:
1. **Baseline**: Write a clean, portable scalar C++ implementation.
2. **Measurement**: Profile the code and analyze compiler optimizations (from `-O0` to `-Ofast`) to identify bottlenecks.
3. **Optimization**: Rewrite the identified "hot" kernels using RISC-V Vector (RVV) intrinsics.
4. **Verification**: Prove with data that the optimized vector code produces the exact same output as the C++ baseline across different Vector Lengths (VLEN).

### Technical Stack
- **Host:** WSL2/Ubuntu 24.04 or Arch Linux
- **Shell:** Bash / Fish
- **Compiler:** `riscv64-unknown-elf-g++`
- **Emulator:** QEMU (RVV 1.0 support)
- **Testing:** GoogleTest (Host) & Assert-based (Target)
- **Documentation:** Doxygen

---

## Visual Guides

### 1. Project Mindmap & Mastery Path
This diagram breaks down the core skills required to master this project, mapping out the relationships between the toolchain, C++ engineering, computer vision algorithms, testing, and RVV optimization.

<img width="1408" height="768" alt="Gemini_Generated_Image_bmyjxbmyjxbmyjxb" src="https://github.com/user-attachments/assets/feca7eb9-67f5-496c-a534-8a2e8c31e6b1" />


### 2. Toolchain & Emulation Workflow
This visualization shows how the dual-target workflow connects your host machine to the RISC-V emulation environment, detailing the path from source code to functional correctness and performance profiling.

<img width="1380" height="752" alt="Gemini_Generated_Image_6ov3qr6ov3qr6ov3" src="https://github.com/user-attachments/assets/27a997f9-5db9-4e65-8d5e-502c09896a57" />



---

## Project Phases & Deliverables

The workload is distributed across the following core phases:
* **Setup:** Building the RISC-V GCC toolchain (with `--with-arch=rv64gcv`) and QEMU from source.
* **Baseline:** Implementing image I/O, blur, and Sobel computations using generic C++ templates.
* **Testing:** Writing GoogleTest unit tests for host and assert-based equivalence tests for QEMU.
* **Compilers & Profiling:** Sweeping compiler flags, profiling per-stage execution times, and reading auto-vectorization reports.
* **Vectorization:** Rewriting slow kernels with RVV intrinsics (handling strip-mining, LMUL selection, and data widening).
* **Reporting & Demo:** Producing an optimization report, an AI Usage Log, and preparing for a live technical demonstration.

---

## Digital Electronics Context (Hardware meets Software)
While executed on a software emulator (QEMU), the RVV optimizations in this project directly command simulated digital hardware structures:

* **The Vector Datapath (VPUs):** Using intrinsics like `vadd` commands multiple ALUs (lanes) in parallel. Instead of processing one pixel per clock cycle, the hardware processes batches simultaneously.
* **Vector Register File (VRF) & VLEN:** Sweeping `VLEN` tests the software against different hardware register widths. High VLEN means massive registers (e.g., 512 bits) managed by complex flip-flop arrays.
* **Length Multiplier (LMUL):** Grouping registers (e.g., LMUL=8) forces hardware control logic to multiplex multiple registers into massive, continuous data chunks for the ALUs.
* **Vector Length Agnostic (VLA) Logic:** Using `vsetvli` interacts with the `vl` Control and Status Register (CSR). The digital logic uses masking (AND gates) to dynamically enable/disable ALU lanes based on the hardware's physical capacity, ensuring safe execution across any VLEN.

---
### Project Structure
```plaintext
.
├── assets/             # Input/Output raw images (width*height bytes)
├── build/              # Build artifacts (host/ for x86, target/ for RISC-V)
├── include/            # Header files (.hpp)
├── src/                # Implementation files (.cpp)
├── scripts/            # Automated scripts
├── tests/              # GoogleTest (host) and RVV equivalence tests
├── tools/              # Custom profiling or visualization tools
├── Makefile            # Dual-target build system
└── Doxyfile            # Doxygen configuration
```

---

## Getting Started

### Environment Setup
The setup process is fully automated for Ubuntu 24.04 & Arch Linux users.
**The automation handles:**
1. **System Dependencies:** Compilers, libraries, and build tools.
2. **RISC-V GNU Toolchain:** `riscv64-unknown-elf-` with RVV support.
3. **QEMU User-Mode:** `qemu-riscv64` for vector emulation.
4. **GoogleTest:** Native Host Support for logic validation.
5. **Project Workspace:** Initializing the professional folder hierarchy.

Some of the tools in the setup runs only on Linux; thus, for Windows OS users WSL2 (Windows Subsystem for Linux) with Ubuntu 24.04 is needed. WSL2 provides a full Linux kernel inside Windows and is the standard approach used in industry for cross platform embedded development.

<img width="1408" height="768" alt="Gemini_Generated_Image_198pgn198pgn198p" src="https://github.com/user-attachments/assets/2523c1cb-5f8b-4927-b18c-31960bef8a6e" />


**Run in terminal:**
```bash
chmod +x scripts/phase_one_setup.sh
./scripts/phase_one_setup.sh
source ~/.bashrc
```

---

## Build System (Makefile)

The project utilizes a **Dual-Target Makefile** to bridge the gap between development and hardware optimization.

### Available Commands
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

### Configuration Overrides
You can override hardware parameters directly from the command line:

* **Vector Length (VLEN):** Test scaling by changing the simulated bit-width (must be a power of 2).
  ```bash
  make run VLEN=128  # Simulate lower-end hardware
  make run VLEN=256  # Simulate mid-range hardware
  ```
* **Hardware Profile:** QEMU execution is configured to use `-cpu max,v=true` to ensure full Vector 1.0 specification compliance.

---

## Authors
- Abdelrhman Atta
- Aly Gano
- Marawan Mohamed
- Mouaz Kfafy
- Mohamed elHosary

## Resources & References
- [RISC-V Vector Quick Intro](https://blog.timhutt.co.uk/riscv-vector/#the-end)
- [RISCV-V RVV Intrinsics Viewer](https://dzaima.github.io/intrinsics-viewer/)
- [Doxygen Documentation](https://www.doxygen.nl/)
- [Markdown Guide](https://www.markdownguide.org/cheat-sheet/)
- [Cornell University: RISC-V Vector Extension](https://www.cs.cornell.edu/courses/cs6120/2023fa/blog/rvv-llvm-gisel/)
- [Gaussian Blue](https://computergraphics.stackexchange.com/questions/39/how-is-gaussian-blur-implemented)
