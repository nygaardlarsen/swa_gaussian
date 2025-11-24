#!/bin/bash
set -e

echo "=== Updating pip ==="
pip install --upgrade pip

echo "=== Installing required Python packages ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib tqdm pandas
pip install gpytorch
pip install tabulate

echo "=== Cloning SWAG-Gaussian repository ==="
if [ ! -d swa_gaussian ]; then
    git clone https://github.com/nygaardlarsen/swa_gaussian.git
fi

cd swa_gaussian

echo "=== Installing repo (setup.py develop) ==="
pip install -r requirements.txt || true   # requirements.txt is old; ignore minor errors
python setup.py develop

echo "=== Creating data folder ==="
mkdir -p /work/data/cifar

echo "=== init.sh complete ==="
