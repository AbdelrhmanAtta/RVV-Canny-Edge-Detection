HOST_CXX = g++
RV_CXX   	:= riscv64-linux-gnu-g++
GTEST    	:= $(HOME)/googletest-installed
SRCS        := $(wildcard src/*.cpp)
LIB_SRCS    := $(filter-out src/main.cpp, $(SRCS))

RV_FLAGS = -std=c++23 -march=rv64gcv -O3 -static -Iinc
RV_FLAGS_BASE = -std=c++23 -march=rv64gcv -static -Iinc
HOST_FLAGS = -std=c++23 -O3 -I$(GTEST)/include -L$(GTEST)/lib -lgtest -lgtest_main -lpthread

# Targets
.PHONY: all clean run test run-test list-tests \
        bench_O0 bench_O2 bench_O3 bench_Os bench_Ofast bench_all \
        autovec_report size_report

all: canny_rv test


# Main Canny Pipeline Build
canny_rv: $(SRCS)
	@mkdir -p ./build/target/release
	$(RV_CXX) $(RV_FLAGS) $(SRCS) -o build/target/release/canny_rv.elf

# Standard Host Testing (GoogleTest)
test: tests/host_tests.cpp
	@mkdir -p ./build/host/debug
	$(HOST_CXX) -DHOST_MODE tests/host_tests.cpp $(HOST_FLAGS) -o build/host/debug/unit_tests
	./build/host/debug/unit_tests

# Run the main pipeline
run: canny_rv
	qemu-riscv64 -cpu max,vlen=512 ./build/target/release/canny_rv.elf

# Cleanup build artifacts
clean:
	rm -rf build/*


# AUTOMATIC PATTERN RULES FOR QUICK TESTING

# Compiles any .cpp in tests/ to an .elf in build/target/
build/target/debug/%.elf: tests/%.cpp $(LIB_SRCS)
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_FLAGS) $^ -o $@

# Run any test by name: make run-tests NAME=sanity
run-tests: build/target/debug/$(NAME).elf
	qemu-riscv64 -cpu max,vlen=512 build/target/debug/$(NAME).elf

# Utility to see what you can run
list-tests:
	@ls tests/*.cpp | xargs -n 1 basename | sed 's/\.cpp//'


# ── PHASE 4: BENCHMARKING ──────────────────────────────────────

bench_O0: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_FLAGS_BASE) -O0 $(SRCS) -o build/bench/canny_O0.elf
	@echo "=== -O0 ==="
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_O0.elf

bench_O2: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_FLAGS_BASE) -O2 $(SRCS) -o build/bench/canny_O2.elf
	@echo "=== -O2 ==="
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_O2.elf

bench_O3: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_FLAGS_BASE) -O3 $(SRCS) -o build/bench/canny_O3.elf
	@echo "=== -O3 ==="
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_O3.elf

bench_Os: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_FLAGS_BASE) -Os $(SRCS) -o build/bench/canny_Os.elf
	@echo "=== -Os ==="
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_Os.elf

bench_Ofast: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_FLAGS_BASE) -Ofast $(SRCS) -o build/bench/canny_Ofast.elf
	@echo "=== -Ofast ==="
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_Ofast.elf

# Run all benchmarks sequentially
bench_all: bench_O0 bench_O2 bench_O3 bench_Os bench_Ofast

# Auto-vectorization report (saved to results/)
autovec_report: $(SRCS)
	@mkdir -p ./build/bench ./results
	$(RV_CXX) $(RV_FLAGS_BASE) -O3 -fopt-info-vec-all $(SRCS) \
	    -o build/bench/canny_autovec.elf 2> results/autovec_report.txt
	@echo "Saved to results/autovec_report.txt"

# Binary size comparison
size_report:
	@echo "=== Binary sizes ==="
	@ls -lh build/bench/*.elf 2>/dev/null || echo "Run make bench_all first"