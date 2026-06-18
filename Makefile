HOST_CXX = g++
RV_CXX      := riscv64-linux-gnu-g++
GTEST       := $(HOME)/googletest-installed
SRCS        := $(wildcard src/*.cpp)
LIB_SRCS    := $(filter-out src/main.cpp, $(SRCS))

RV_CXXFLAGS = -std=c++23 -march=rv64gcv -O3 -static -Iinc
RV_CXXFLAGS_BASE = -std=c++23 -march=rv64gcv -static -Iinc
HOST_CXXFLAGS = -std=c++23 -O3 -I$(GTEST)/include -L$(GTEST)/lib -lgtest -lgtest_main -lpthread

# Targets
.PHONY: all clean run tests run-tests list-tests \
        bench-O0 bench-O2 bench-O3 bench-Os bench-Ofast bench-all \
        autovec-report size-report

all: canny-rv tests


# Main Canny Pipeline Build
canny-rv: $(SRCS)
	@mkdir -p ./build/target/release
	$(RV_CXX) $(RV_CXXFLAGS) $(SRCS) -o build/target/release/canny_rv.elf

# Standard Host Testing (GoogleTest)
tests: tests/unit_tests.cpp $(LIB_SRCS)
	@mkdir -p ./build/host/debug
	$(HOST_CXX) -DHOST_MODE $^ $(HOST_CXXFLAGS) -o build/host/debug/unit_tests
	./build/host/debug/unit_tests

# Run the main pipeline
run: canny-rv
	qemu-riscv64 -cpu max,vlen=512 ./build/target/release/canny_rv.elf

# Cleanup build artifacts
clean:
	rm -rf build/*


# Compiles any .cpp in tests/ to an .elf in build/target/
build/target/debug/%.elf: tests/%.cpp $(LIB_SRCS)
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_CXXFLAGS) $^ -o $@

# Run any tests by name: make run-tests NAME=sanity
run-tests: build/target/debug/$(NAME).elf
	qemu-riscv64 -cpu max,vlen=512 build/target/debug/$(NAME).elf

# Utility to see what you can run
list-tests:
	@ls tests/*.cpp | xargs -n 1 basename | sed 's/\.cpp//'


# Benchmarking Targets
bench-O0: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_BASE) -O0 $(SRCS) -o build/bench/canny_O0.elf
	@echo "-O0"
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_O0.elf

bench-O2: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_BASE) -O2 $(SRCS) -o build/bench/canny_O2.elf
	@echo "-O2"
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_O2.elf

bench-O3: $(SRCS)
	@mkdir -p ./build/bench
	$(RV_CXX) $(RV_CXXFLAGS_BASE) -O3 $(SRCS) -o build/bench/canny_O3.elf
	@echo "-O3"
	qemu-riscv64 -cpu max,vlen=512 build/bench/canny_O3.elf

# Run all benchmarks sequentially
bench-all: bench-O0 bench-O2 bench-O3

# Auto-vectorization report (saved to results/)
autovec-report: $(SRCS)
	@mkdir -p ./build/bench ./results
	$(RV_CXX) $(RV_CXXFLAGS_BASE) -O3 -fopt-info-vec-all $(SRCS) \
		-o build/bench/canny_autovec.elf 2> results/autovec_report.txt
	@echo "Saved to results/autovec_report.txt"

# Binary size comparison
size-report:
	@echo "Binary sizes"
	@ls -lh build/bench/*.elf 2>/dev/null || echo "Run make bench-all first"