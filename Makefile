HOST_CXX = g++
RV_CXX   = riscv64-unknown-elf-g++
GTEST    = $(HOME)/googletest-installed


RV_FLAGS   = -march=rv64gcv -O3 -static
HOST_FLAGS = -O3 -I$(GTEST)/include -L$(GTEST)/lib -lgtest -lgtest_main -lpthread

# Targets
.PHONY: all clean run test run-test list-tests

all: canny_rv test


# Main Canny Pipeline Build
canny_rv: src/main.cpp
	@mkdir -p ./build/target/release
	$(RV_CXX) $(RV_FLAGS) src/main.cpp -o build/target/release/canny_rv.elf

# Standard Host Testing (GoogleTest)
test: tests/host_tests.cpp
	@mkdir -p ./build/host/debug
	$(HOST_CXX) -DHOST_MODE tests/host_tests.cpp $(HOST_FLAGS) -o build/host/debug/unit_tests
	./build/host/debug/unit_tests

# Run the main pipeline
run: canny_rv
	qemu-riscv64 -cpu max,vlen=512 build/target/release/canny_rv.elf

# Cleanup build artifacts
clean:
	rm -rf build/*


# AUTOMATIC PATTERN RULES FOR QUICK TESTING

# Compiles any .cpp in tests/ to an .elf in build/target/
build/target/debug/%.elf: tests/%.cpp
	@mkdir -p ./build/target/debug
	$(RV_CXX) $(RV_FLAGS) $< -o $@

# Run any test by name: make run-test NAME=sanity
run-test: build/target/debug/$(NAME).elf
	qemu-riscv64 -cpu max,vlen=512 build/target/debug/$(NAME).elf

# Utility to see what you can run
list-tests:
	@ls tests/*.cpp | xargs -n 1 basename | sed 's/\.cpp//'
