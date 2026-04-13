#!/bin/bash

# Header
echo "
  _______  ___________  ___________  _______
 /  _   | /          / /          / /  _   |
/  /_|  |/___  ____ / /___  ____ / /  /_|  |
|   _   |   /  /         /  /      |   _   |
|  | |  |  /  /         /  /       |  | |  |
|__| |__| /__/         /__/        |__| |__|

      RISC-V VECTOR TOOLCHAIN AUTOMATOR
      Atta bymse 3leko <3
"
sleep 3


# Config
TOOLCHAIN_DIR="$HOME/riscv-toolchain"
# Note: Separating install dir from source dir to prevent CMake errors
GTEST_INSTALL_DIR="$HOME/googletest-installed"
PROJECT_TITLE="RVV-Canny-Edge-Detection"
JOBS=$(nproc)

echo -e "\n-------------------------------------------------------"
echo "Installation starting on $(uname -n)..."
echo -e "-------------------------------------------------------\n"
sleep 3


# System Dependencies
echo -e "\n-------------------------------------------------------"
echo "Installing system dependencies..."
echo -e "-------------------------------------------------------\n"
sleep 1
sudo apt update
sudo apt install -y autoconf automake build-essential bison flex texinfo \
    gperf libtool patchutils bc git cmake ninja-build \
    libglib2.0-dev libpixman-1-dev libslirp-dev \
    libmpc-dev libmpfr-dev libgmp-dev zlib1g-dev libexpat-dev python3


# RISCV Cross Toolchain
if [ ! -d "$HOME/riscv-gnu-toolchain" ]; then
    echo -e "\n-------------------------------------------------------"
    echo "Cloning RISC-V GNU toolchain..."
    echo -e "-------------------------------------------------------\n"
    sleep 1
    git clone --recursive --depth 1 https://github.com/riscv-collab/riscv-gnu-toolchain
else
    echo -e "\n-------------------------------------------------------"
    echo "RISC-V GNU toolchain already exists, skipping clone."
    echo -e "-------------------------------------------------------\n"
    sleep 1
fi

cd "$HOME/riscv-gnu-toolchain"
echo -e "\n-------------------------------------------------------"
echo "Configuring RISC-V GNU toolchain..."
echo -e "-------------------------------------------------------\n"
sleep 1
./configure --prefix="$TOOLCHAIN_DIR" --with-arch=rv64gcv --with-abi=lp64d

echo -e "\n-------------------------------------------------------"
echo -e "Building RISC-V GNU toolchain with $JOBS jobs...\nThis may take a while, study fields till its done :)"
echo -e "-------------------------------------------------------\n"
sleep 3
make -j"$JOBS"


# Add toolchain path in .bashrc
if [[ ":$PATH:" != *":$TOOLCHAIN_DIR/bin:"* ]]; then
    echo -e "\n-------------------------------------------------------"
    echo "Adding toolchain to PATH in .bashrc..."
    echo "export PATH=\"\$PATH:$TOOLCHAIN_DIR/bin\"" >> ~/.bashrc
    echo -e "-------------------------------------------------------\n"
    sleep 1
    export PATH="$PATH:$TOOLCHAIN_DIR/bin"
else
    echo -e "\n-------------------------------------------------------"
    echo "Toolchain path already in PATH, skipping .bashrc modification."
    echo -e "-------------------------------------------------------\n"
    sleep 1
fi

if "$TOOLCHAIN_DIR/bin/riscv64-unknown-elf-gcc" --version; then
    echo -e "\n-------------------------------------------------------"
    echo "SUCCESS: RISC-V Toolchain is operational"
    echo -e "-------------------------------------------------------\n"
else
    echo -e "\n-------------------------------------------------------"
    echo "ERROR: RISC-V Toolchain installation failed"
    echo -e "-------------------------------------------------------\n"
    exit 1
fi
sleep 1


# Bulding QEMU
cd "$HOME"
if [ ! -d "$HOME/qemu" ]; then
    echo -e "\n-------------------------------------------------------"
    echo "Cloning QEMU repo..."
    echo -e "-------------------------------------------------------\n"
    sleep 1
    git clone --depth 1 https://github.com/qemu/qemu
else
    echo -e "\n-------------------------------------------------------"
    echo "QEMU repo already exists, skipping clone."
    echo -e "-------------------------------------------------------\n"
    sleep 1
fi

cd qemu
mkdir -p build && cd build
echo -e "\n-------------------------------------------------------"
echo "Configuring QEMU for RISC-V..."
echo -e "-------------------------------------------------------\n"
sleep 1
../configure --target-list=riscv64-linux-user --enable-plugins
echo -e "\n-------------------------------------------------------"
echo "Building QEMU..."
echo -e "-------------------------------------------------------\n"
make -j"$JOBS"
sudo make install


# Building Google Test
cd "$HOME"
if [ ! -d "googletest" ]; then
    echo -e "\n-------------------------------------------------------"
    echo "Cloning Google Test repo..."
    echo -e "-------------------------------------------------------\n"
    sleep 1
    git clone --depth 1 https://github.com/google/googletest.git
else
    echo -e "\n-------------------------------------------------------"
    echo "Google Test repo already exists, skipping clone."
    echo -e "-------------------------------------------------------\n"
    sleep 1
fi

cd googletest
mkdir -p build && cd build
echo -e "\n-------------------------------------------------------"
echo "Configuring Google Test..."
echo -e "-------------------------------------------------------\n"
sleep 1
cmake .. -DCMAKE_INSTALL_PREFIX="$GTEST_INSTALL_DIR"
echo -e "\n-------------------------------------------------------"
echo "Building and Installing Google Test..."
echo -e "-------------------------------------------------------\n"
make -j"$JOBS"
make install


echo -e "\n-------------------------------------------------------"
echo "Creating project structure..."
echo -e "-------------------------------------------------------\n"
sleep 1

# Professional Architecture Folders
mkdir -p "$HOME/$PROJECT_TITLE/assets"
mkdir -p "$HOME/$PROJECT_TITLE/include"
mkdir -p "$HOME/$PROJECT_TITLE/src"
mkdir -p "$HOME/$PROJECT_TITLE/tests"
mkdir -p "$HOME/$PROJECT_TITLE/tools"
mkdir -p "$HOME/$PROJECT_TITLE/build/host"
mkdir -p "$HOME/$PROJECT_TITLE/build/target"

cat <<'EOF' > "$HOME/$PROJECT_TITLE/Makefile"
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
EOF

echo "-------------------------------------------------------"
echo "DONE!"
echo "Workspace: ~/$PROJECT_TITLE"
echo "Run 'source ~/.bashrc' to enable the tools."
echo "-------------------------------------------------------"