# Toolchain & Paths
HOST_CXX          := g++
RV_CXX            := riscv64-linux-gnu-g++
GEM5_BIN          := $(HOME)/gem5/build/RISCV/gem5.opt
GEM5_SCRIPT       := ./tools/run_gem5.py
GTEST             := $(HOME)/googletest-installed
GTEST_RV          := $(HOME)/googletest-riscv
GEM5_INCLUDE      := $(HOME)/gem5/include
GEM5_LIB          := $(HOME)/gem5/util/m5/build/riscv/out

# Flags
RV_CXXFLAGS_BASE  := -std=c++23 -static -Iinc -DGEM5_MODE -I$(GEM5_INCLUDE)
RV_CXXFLAGS_RVV   := $(RV_CXXFLAGS_BASE) -march=rv64gcv
RV_CXXFLAGS_SCALAR:= $(RV_CXXFLAGS_BASE) -march=rv64gc
RV_LDFLAGS        := -L$(GEM5_LIB) -lm5
HOST_CXXFLAGS     := -std=c++23 -Iinc -I$(GTEST)/include -L$(GTEST)/lib -lgtest -lgtest_main -pthread

# Files
SRCS              := $(wildcard src/*.cpp)
LIB_SRCS          := $(filter-out src/main.cpp, $(SRCS))
TARGET            := build/target/release/canny_rv.elf

.PHONY: all clean run gem5-run tests bench-all size-report list-tests autovec-report \
        bench-rvv-O0 bench-rvv-O2 bench-rvv-O3 \
        bench-scalar-O0 bench-scalar-O2 bench-scalar-O3 \
        vla-equiv qemu-tests

all: $(TARGET)

# Main Pipeline Build
# NO -DTESTS here. This ensures `make run` uses production paths.
$(TARGET): $(SRCS)
	@mkdir -p ./build/target/release
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -O3 $(SRCS) -o $@ $(RV_LDFLAGS)

# Functional Testing (QEMU)
run: $(TARGET)
	qemu-riscv64 -cpu rv64,v=true,vlen=128,elen=64 $(TARGET)

# Run standard tests on QEMU using RISC-V GoogleTest
qemu-tests: tests/host_tests.cpp $(LIB_SRCS)
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -I$(GTEST_RV)/include -L$(GTEST_RV)/lib $^ -o build/target/debug/host_tests.elf $(RV_LDFLAGS) -lgtest -lgtest_main -pthread
	@echo "\n>>> Running Pipeline Tests on QEMU (VLEN=512)..."
	qemu-riscv64 -cpu max,vlen=512 build/target/debug/host_tests.elf

# Run Equivalence Tests to verify Vector-Length Agnostic (VLA) compliance
vla-equiv: tests/host_tests.cpp $(LIB_SRCS)
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -I$(GTEST_RV)/include -L$(GTEST_RV)/lib $^ -o build/target/debug/host_tests.elf $(RV_LDFLAGS) -lgtest -lgtest_main -pthread
	@echo "\n>>> Running Equivalence Tests with VLEN=128"
	qemu-riscv64 -cpu max,vlen=128 build/target/debug/host_tests.elf
	@echo "\n>>> Running Equivalence Tests with VLEN=256"
	qemu-riscv64 -cpu max,vlen=256 build/target/debug/host_tests.elf
	@echo "\n>>> Running Equivalence Tests with VLEN=512"
	qemu-riscv64 -cpu max,vlen=512 build/target/debug/host_tests.elf

# Architectural Profiling (gem5)
# Usage: make gem5-run NAME=your_test
gem5-run: build/target/debug/$(NAME).elf
	@echo "Running $(NAME) in gem5..."
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=$<
	@echo "Simulation complete. Check m5out/stats.txt for accurate cycles."

# Standard Host Testing
tests: tests/unit_tests.cpp $(LIB_SRCS)
	@mkdir -p ./build/host/debug
	$(HOST_CXX) -DHOST_MODE -DTESTS $^ $(HOST_CXXFLAGS) -o build/host/debug/unit_tests
	./build/host/debug/unit_tests

# RVV Benchmarks
bench-rvv-O0: ./tests/benchmark.cpp $(LIB_SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -O0 $^ -o build/bench/canny_rvv_O0.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=build/bench/canny_rvv_O0.elf

bench-rvv-O2: ./tests/benchmark.cpp $(LIB_SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -O2 $^ -o build/bench/canny_rvv_O2.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=build/bench/canny_rvv_O2.elf

bench-rvv-O3: ./tests/benchmark.cpp $(LIB_SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -O3 $^ -o build/bench/canny_rvv_O3.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=build/bench/canny_rvv_O3.elf

# Scalar Benchmarks
bench-scalar-O0: ./tests/benchmark.cpp $(LIB_SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_SCALAR) -DTESTS -O0 $^ -o build/bench/canny_scalar_O0.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=build/bench/canny_scalar_O0.elf

bench-scalar-O2: ./tests/benchmark.cpp $(LIB_SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_SCALAR) -DTESTS -O2 $^ -o build/bench/canny_scalar_O2.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=build/bench/canny_scalar_O2.elf

bench-scalar-O3: ./tests/benchmark.cpp $(LIB_SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_SCALAR) -DTESTS -O3 $^ -o build/bench/canny_scalar_O3.elf $(RV_LDFLAGS)
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=build/bench/canny_scalar_O3.elf

# Convenience groups
bench-rvv:    bench-rvv-O0    bench-rvv-O2    bench-rvv-O3
bench-scalar: bench-scalar-O0 bench-scalar-O2 bench-scalar-O3
bench-all:    bench-rvv bench-scalar

# Report
size-report:
	@echo "Binary sizes:"
	@ls -lh build/bench/*.elf 2>/dev/null || echo "Run make bench-all first"

# Cleanup
clean:
	rm -rf build/* m5out/ results/

# Generic debug elf builder
build/target/debug/%.elf: tests/%.cpp $(LIB_SRCS)
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_CXXFLAGS_RVV) -DTESTS -O2 $^ -o $@ $(RV_LDFLAGS)

run-tests: build/target/debug/$(NAME).elf
	$(GEM5_BIN) $(GEM5_SCRIPT) --cmd=$<

list-tests:
	@ls tests/*.cpp | xargs -n 1 basename | sed 's/\.cpp//'