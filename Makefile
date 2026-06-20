# ============================================================================
# Toolchain & Paths
# ============================================================================
HOST_CXX          := g++
RV_CXX            := riscv64-linux-gnu-g++
GEM5_BIN          := $(HOME)/gem5/build/RISCV/gem5.opt
GEM5_SCRIPT       := ./tools/run_gem5.py
GTEST             := $(HOME)/googletest-installed
GTEST_RV          := $(HOME)/googletest-riscv
GEM5_INCLUDE      := $(HOME)/gem5/include
GEM5_LIB          := $(HOME)/gem5/util/m5/build/riscv/out

# ============================================================================
# Compiler Flags
# ============================================================================
RV_CXXFLAGS_BASE  := -std=c++23 -static -Iinc -DGEM5_MODE -I$(GEM5_INCLUDE)
RV_CXXFLAGS_RVV   := $(RV_CXXFLAGS_BASE) -march=rv64gcv
RV_CXXFLAGS_SCALAR:= $(RV_CXXFLAGS_BASE) -march=rv64gc
RV_LDFLAGS        := -L$(GEM5_LIB) -lm5

HOST_CXXFLAGS     := -std=c++23 -Iinc -I$(GTEST)/include -L$(GTEST)/lib -lgtest -lgtest_main -pthread

# ============================================================================
# Source Files
# ============================================================================
SRCS              := $(wildcard src/*.cpp)
LIB_SRCS          := $(filter-out src/main.cpp, $(SRCS))

# ============================================================================
# Main Target (production ELF)
# ============================================================================
TARGET            := build/target/release/canny_rv.elf
.PHONY: all
all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p ./build/target/release
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -O3 $(SRCS) -o $@ $(RV_LDFLAGS)

# ============================================================================
# Running on QEMU (functional test)
# ============================================================================
.PHONY: run
run: $(TARGET)
	qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 $(TARGET)

# ============================================================================
# GoogleTest Unit Tests (QEMU)
# ============================================================================
QEMU_TEST_ELF := build/target/debug/qemu_tests.elf

.PHONY: qemu-tests
qemu-tests: tests/qemu_tests.cpp $(LIB_SRCS)
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -I$(GTEST_RV)/include -L$(GTEST_RV)/lib $^ -o $(QEMU_TEST_ELF) $(RV_LDFLAGS) -lgtest -lgtest_main -pthread
	@echo "\n>>> Running Pipeline Tests on QEMU (VLEN=128)..."
	qemu-riscv64 -cpu max,vlen=128 $(QEMU_TEST_ELF) || exit 1
	@echo "\n>>> Running Pipeline Tests on QEMU (VLEN=256)..."
	qemu-riscv64 -cpu max,vlen=256 $(QEMU_TEST_ELF) || exit 1
	@echo "\n>>> Running Pipeline Tests on QEMU (VLEN=512)..."
	qemu-riscv64 -cpu max,vlen=512 $(QEMU_TEST_ELF) || exit 1
	@echo "\nAll VLEN tests passed."

# ============================================================================
# Host (x86) Unit Tests
# ============================================================================
.PHONY: tests
tests: tests/unit_tests.cpp $(LIB_SRCS)
	@mkdir -p ./build/host/debug
	$(HOST_CXX) -DHOST_MODE -DTESTS $^ $(HOST_CXXFLAGS) -o build/host/debug/unit_tests
	./build/host/debug/unit_tests

# ============================================================================
# Benchmarking (gem5) – grouped by optimisation level
# ============================================================================
BENCH_SRC := ./tests/benchmark.cpp
BENCH_DIR := ./build/bench

# Pattern for RVV benchmarks: bench-rvv-<OPT>
.PHONY: bench-rvv-%
bench-rvv-%: $(BENCH_SRC) $(LIB_SRCS)
	@mkdir -p $(BENCH_DIR)
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -$* $^ -o $(BENCH_DIR)/canny_rvv_$*.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=$(BENCH_DIR)/canny_rvv_$*.elf

# Pattern for scalar benchmarks: bench-scalar-<OPT>
.PHONY: bench-scalar-%
bench-scalar-%: $(BENCH_SRC) $(LIB_SRCS)
	@mkdir -p $(BENCH_DIR)
	$(RV_CXX) $(RV_CXXFLAGS_SCALAR) -DTESTS -$* $^ -o $(BENCH_DIR)/canny_scalar_$*.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=$(BENCH_DIR)/canny_scalar_$*.elf

# Convenience groups
.PHONY: bench-rvv bench-scalar bench-all
bench-rvv:    bench-rvv-O0 bench-rvv-O2 bench-rvv-O3
bench-scalar: bench-scalar-O0 bench-scalar-O2 bench-scalar-O3 bench-scalar-Ofast bench-scalar-Os
bench-all:    bench-rvv bench-scalar

# ============================================================================
# Individual gem5 run for any test (requires NAME=...)
# ============================================================================
.PHONY: gem5-run
gem5-run: build/target/debug/$(NAME).elf
	@echo "Running $(NAME) in gem5..."
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=$<
	@echo "Simulation complete. Check m5out/stats.txt for accurate cycles."

# Pattern to build any test ELF (e.g., make build/target/debug/foo.elf)
build/target/debug/%.elf: tests/%.cpp $(LIB_SRCS)
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -O2 $^ -o $@ $(RV_LDFLAGS)

# ============================================================================
# Utility / Info
# ============================================================================
.PHONY: list-tests
list-tests:
	@ls tests/*.cpp | xargs -n 1 basename | sed 's/\.cpp//'

.PHONY: size-report
size-report: $(TARGET)
	@echo "Size report for $(TARGET):"
	@riscv64-linux-gnu-size $(TARGET)

.PHONY: autovec-report
autovec-report: $(TARGET)
	@echo "Auto-vectorisation report (RVV):"
	@riscv64-linux-gnu-objdump -d $(TARGET) | grep -E "vset|vle|vse|vadd|vmul" | wc -l

# ============================================================================
# Cleanup
# ============================================================================
.PHONY: clean
clean:
	rm -rf build/* m5out/ results/
	