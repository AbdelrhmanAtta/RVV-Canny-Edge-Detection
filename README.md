# Canny Edge Detection on RISC-V using RVV
**Embedded Systems Project Target Architecture:** `rv64gcv` (QEMU User-mode)

## Overview
This project implements a high-performance **Canny edge detection** pipeline. The core objective is the migration from a clean C++ scalar baseline to a hand-optimized **RVV (RISC-V Vector)** intrinsic implementation.

### Pipeline Stages
1. **Gaussian Blur:** 5x5 Kernel (2D Convolution)
2. **Sobel Operator:** $G_x$ and $G_y$ gradients (3x3 Kernels)
3. **Magnitude & Direction:** $L_1/L_2$ Norms and 4-way direction quantization.
4. **NMS & Hysteresis:** (Bonus) Non-Maximum Suppression and Double Thresholding.

### Technical Stack
- **Host:** WSL2/Ubuntu 24.04 or Arch Linux
- **Shell:** Bash / Fish
- **Compiler:** `riscv64-unknown-elf-g++`
- **Emulator:** QEMU (RVV 1.0 support)
- **Testing:** GoogleTest (Host) & Assert-based (Target)
- **Documentation:** Doxygen

---

### Project Structure
```plaintext
.
├── assets/             # Input/Output raw images (width*height bytes)
├── build/              # Build artifacts (host/ for x86, target/ for RISC-V)
├── include/            # Header files (.hpp)
├── src/                # Implementation files (.cpp)
<<<<<<< HEAD
├── scripts/            # Automated scripts
=======
├── scripts/            # Environment setup (Arch/Ubuntu) and automation
>>>>>>> 278457a (build: overhaul build system for dual-target RVV development)
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

## Resources
- [RISC-V Vector Quick Intro](https://blog.timhutt.co.uk/riscv-vector/#the-end)
- [RISCV-V RVV Intrinsics Viewer](https://dzaima.github.io/intrinsics-viewer/)
- [Doxygen Documentation](https://www.doxygen.nl/)
- [Markdown Guide](https://www.markdownguide.org/cheat-sheet/)