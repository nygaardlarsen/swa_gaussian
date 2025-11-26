#!/bin/bash
# ========================================
# init.sh - init script for SWAG-Gaussian
# ========================================

# Stop on errors
set -e

# --- Environment setup ---
echo "Setting up environment..."

# export PYTHONPATH="/work/SWAG:$PYTHONPATH"
# echo "PYTHONPATH set to $PYTHONPATH"

# --- Update pip ---
echo "Updating pip..."
python3 -m pip install --upgrade --user pip

# --- Install required Python packages ---
echo "Installing Python packages..."
# Brug brugerinstallation for at undgå permissions-problemer
python3 -m pip install --user torch torchvision matplotlib tqdm pandas gpytorch tabulate scikit-learn torchmetrics

echo "Running setup.py..."

python3 -m pip install --user -e /work/SWAG

# --- Optional: vis hvilke pakker der er installeret ---
echo "Installed Python packages:"
python3 -m pip list --user

mkdir -p /work/SWAG/data/cifar
echo "Data directory created: /work/SWAG/data/cifar"

echo "Initialization done."
