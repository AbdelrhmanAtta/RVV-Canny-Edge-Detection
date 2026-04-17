#!/bin/bash

# Header
echo "
  _______  ___________  ___________  _______
 /  _   | /          / /          / /  _   |
/  /_|  |/___  ____ / /___  ____ / /  /_|  |
|   _   |   /  /         /  /      |   _   |
|  | |  |  /  /         /  /       |  | |  |
|__| |__| /__/         /__/        |__| |__|

      PYTHON ENVIRONMENT SETUP
      Atta bymse 3leko <3
"
sleep 3


# Config
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

# OS-specific checks and configurations
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ] || [ "$OS" = "pop" ]; then
    PYTHON_CMD="python3"
    # Check for python3-venv
    if ! dpkg -s python3-venv >/dev/null 2>&1; then
        echo "Error: python3-venv is not installed."
        echo "Please run: sudo apt update && sudo apt install python3-venv -y"
        exit 1
    fi
elif [ "$OS" = "arch" ] || [ "$OS" = "manjaro" ]; then
    PYTHON_CMD="python"
else
    echo "Unsupported OS: $OS"
    exit 1
fi

echo -e "\n-------------------------------------------------------"
echo "Setting up python environment..."
echo -e "-------------------------------------------------------\n"
sleep 3


# Creating .venv
$PYTHON_CMD -m venv $VENV_DIR


# Updating pip
echo -e "\n-------------------------------------------------------"
echo "Updating pip"
echo -e "-------------------------------------------------------\n"
sleep 1
./$VENV_DIR/bin/python -m pip install --upgrade pip


# Installing needed packages
echo -e "\n-------------------------------------------------------"
echo "Installing needed packages ($PACKAGES)"
echo -e "-------------------------------------------------------\n"
sleep 1
./$VENV_DIR/bin/python -m pip install $PACKAGES


echo -e "\n-------------------------------------------------------"
echo "DONE!"
echo "Venv: ./$VENV_DIR"
echo "Run 'source $VENV_DIR/bin/activate' to enable the venv."
echo "Run 'python path/to/your_script.py' to run the script." 
echo -e "-------------------------------------------------------\n"