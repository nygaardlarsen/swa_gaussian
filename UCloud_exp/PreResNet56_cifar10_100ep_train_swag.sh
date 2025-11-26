#!/bin/bash
# ========================================
# SWAG-Gaussian run script
# ========================================

set -e  # Stop on errors

# --- Paths ---
DATA_PATH="/work/SWAG/data/cifar"
OUTPUT_DIR="/work/SWAG/models/PreResNet56_cifar10_swag"
RUN_SCRIPT="/work/SWAG/experiments/train/run_swag.py"

# Opret output directory, hvis det ikke findes
mkdir -p "$OUTPUT_DIR"

# --- Kør SWAG træning ---
echo "Running SWAG training..."
python "$RUN_SCRIPT" \
    --dir="$OUTPUT_DIR" \
    --dataset=CIFAR10 \
    --data_path="$DATA_PATH" \
    --model=PreResNet56 \
    --epochs=100 \
    --lr_init=0.1 \
    --wd=3e-4 \
    --swa \
    --swa_start=54 \
    --swa_lr=0.01 \
    --cov_mat \
    --use_test

echo "SWAG training completed."
