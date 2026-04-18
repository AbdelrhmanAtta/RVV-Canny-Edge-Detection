#!/bin/bash

# Header
echo -e "\e[36m
 █████╗ ████████╗████████╗█████╗ 
██╔══██╗╚══██╔══╝╚══██╔══╝██╔══██╗
███████║   ██║      ██║   ███████║
██╔══██║   ██║      ██║   ██╔══██║
██║  ██║   ██║      ██║   ██║  ██║
╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝
 
   RISC-V VECTOR TOOLCHAIN AUTOMATOR
         Atta bymse 3leko <3 \e[0m"
sleep 1

# Config
TOOLCHAIN_DIR="$HOME/riscv-toolchain"
TARGET_CLONE_DIR="$HOME/riscv-gnu-toolchain"
GTEST_INSTALL_DIR="$HOME/googletest-installed"
PROJECT_TITLE="RVV-Canny-Edge-Detection"
JOBS=$(nproc)
VENV_DIR=".venv"
PACKAGES="numpy matplotlib PyQt5"

# OS Detection
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Unable to detect OS from /etc/os-release"
    exit 1
fi

echo "-------------------------------------------------------"
echo "Setup starting on $(uname -n)..."
echo "-------------------------------------------------------"
sleep 1


# System Dependencies
echo "Installing system dependencies..."
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ] || [ "$OS" = "pop" ]; then
    sudo apt update
    sudo apt install -y autoconf automake build-essential bison flex texinfo \
        gperf libtool patchutils bc git cmake ninja-build \
        libglib2.0-dev libpixman-1-dev libslirp-dev \
        libmpc-dev libmpfr-dev libgmp-dev zlib1g-dev libexpat1-dev python3 \
        python3-venv doxygen doxygen-gui doxygen-latex doxygen-doc graphviz \
        libpulse0 libgtk-3-0t64 libasound2t64 libdbus-1-3 \
        libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
        libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0 \
        libqt5gui5t64 riscv64-linux-gnu-gcc riscv64-linux-gnu-binutils \
        riscv64-linux-gnu-glibc
    PYTHON_CMD="python3"
elif [ "$OS" = "arch" ] || [ "$OS" = "manjaro" ]; then
    sudo pacman -Syu --needed --noconfirm \
        base-devel multilib-devel git cmake ninja \
        glib2 pixman libslirp gmp mpc mpfr expat zlib python \
        doxygen graphviz qt5-base texlive-basic texlive-latex \
        riscv64-linux-gnu-gcc riscv64-linux-gnu-binutils \
        riscv64-linux-gnu-glibc
    PYTHON_CMD="python"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

# Force delete if it's not a healthy git repo
if [ -d "$TARGET_CLONE_DIR" ]; then
    if [ ! -d "$TARGET_CLONE_DIR/.git" ]; then
        echo "Found a broken clone at $TARGET_CLONE_DIR, removing it..."
        rm -rf "$TARGET_CLONE_DIR"
    fi
fi

if [ ! -d "$TARGET_CLONE_DIR" ]; then
    echo "Cloning RISC-V GNU toolchain..."
    git clone --recursive --depth 1 --shallow-submodules https://github.com/riscv-collab/riscv-gnu-toolchain "$TARGET_CLONE_DIR"
else
    echo "RISC-V GNU toolchain already exists, skipping clone."
    cd "$TARGET_CLONE_DIR" && git submodule update --init --recursive --depth 1
fi

cd "$TARGET_CLONE_DIR" || { echo "Failed to enter $TARGET_CLONE_DIR"; exit 1; }
./configure --prefix="$TOOLCHAIN_DIR" --with-arch=rv64gcv --with-abi=lp64d

echo "-------------------------------------------------------"
echo "Building RISC-V GNU toolchain with $JOBS jobs...This may take a while, study fields till its done :)"
echo "-------------------------------------------------------"
sleep 1
if [ "$OS" = "arch" ] || [ "$OS" = "manjaro" ]; then
    make -j"$JOBS"
else
    make newlib -j"$JOBS"
fi

# Add toolchain path in .bashrc
if [[ ":$PATH:" != *":$TOOLCHAIN_DIR/bin:"* ]]; then
    echo "export PATH=\"\$PATH:$TOOLCHAIN_DIR/bin\"" >> ~/.bashrc
    export PATH="$PATH:$TOOLCHAIN_DIR/bin"
fi

if [ "$OS" = "arch" ] || [ "$OS" = "manjaro" ]; then
    COMPILER_TEST="$TOOLCHAIN_DIR/bin/riscv64-unknown-elf-gcc"
else
    COMPILER_TEST="$TOOLCHAIN_DIR/bin/riscv64-unknown-elf-g++"
fi
if "$COMPILER_TEST" --version; then
    echo "-------------------------------------------------------"
    echo "SUCCESS: RISC-V Toolchain is operational"
    echo "-------------------------------------------------------"
else
    echo "-------------------------------------------------------"
    echo "ERROR: RISC-V Toolchain installation failed"
    echo "-------------------------------------------------------"
    exit 1
fi

# Bulding QEMU
cd "$HOME"
if [ ! -d "$HOME/qemu" ]; then
    echo "Cloning QEMU repo..."
    git clone --depth 1 https://github.com/qemu/qemu
else
    echo "QEMU repo already exists, skipping clone."
fi

cd qemu
mkdir -p build && cd build
echo "Configuring QEMU for RISC-V..."
../configure --target-list=riscv64-linux-user --enable-plugins
echo "Building QEMU..."
make -j"$JOBS"
sudo make install

# Building Google Test
cd "$HOME"
if [ ! -d "googletest" ]; then
    echo "Cloning Google Test repo..."
    git clone --depth 1 https://github.com/google/googletest.git
else
    echo "Google Test repo already exists, skipping clone."
fi

cd googletest
mkdir -p build && cd build
echo "Configuring Google Test..."
cmake .. -DCMAKE_INSTALL_PREFIX="$GTEST_INSTALL_DIR"
echo "Building and Installing Google Test..."
make -j"$JOBS"
make install

echo "Creating project structure..."
mkdir -p "$HOME/$PROJECT_TITLE/assets"
mkdir -p "$HOME/$PROJECT_TITLE/include"
mkdir -p "$HOME/$PROJECT_TITLE/src"
mkdir -p "$HOME/$PROJECT_TITLE/tests"
mkdir -p "$HOME/$PROJECT_TITLE/tools"
mkdir -p "$HOME/$PROJECT_TITLE/build/host"
mkdir -p "$HOME/$PROJECT_TITLE/build/target"

cd "$HOME/$PROJECT_TITLE" || exit 1
echo "-------------------------------------------------------"
echo "Setting up python environment..."
echo "-------------------------------------------------------"
sleep 1
$PYTHON_CMD -m venv $VENV_DIR
./$VENV_DIR/bin/python -m pip install --upgrade pip
./$VENV_DIR/bin/python -m pip install $PACKAGES

echo "-------------------------------------------------------"
echo "Toolchain Setup"
echo "Workspace: ~/$PROJECT_TITLE"
echo "Run 'source ~/.bashrc' to enable the tools."
echo "-------------------------------------------------------"

echo "-------------------------------------------------------"
echo "Python Setup"
echo "Venv: ./$VENV_DIR"
echo "Run 'source $VENV_DIR/bin/activate' to enable the venv."
echo "Run 'python path/to/your_script.py' to run the script." 
echo "-------------------------------------------------------"
