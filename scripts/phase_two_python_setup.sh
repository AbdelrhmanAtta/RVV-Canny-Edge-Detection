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
PACKAGES="numpy matplotlib"

echo -e "\n-------------------------------------------------------"
echo "Setting up python environment..."
echo -e "-------------------------------------------------------\n"
sleep 3


# Creating .venv
python -m venv $VENV_DIR


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
echo "python path/to/your_script.py to run the script." 
echo -e "-------------------------------------------------------\n"
