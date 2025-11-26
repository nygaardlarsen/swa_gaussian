  #!/bin/bash
# ========================================
# SWAG-Gaussian run script
# ========================================

set -e  # Stop on errors

# --- Paths ---
DATA_PATH="/work/SWAG/data/two_moons"
OUTPUT_DIR="/work/SWAG/models/2dmlp_two_moons_swag"
RUN_SCRIPT="/work/SWAG/experiments/train/run_swag.py"

# Opret output directory, hvis det ikke findes
mkdir -p "$DATA_PATH"
mkdir -p "$OUTPUT_DIR"

# --- Kør SWAG træning ---
echo "Running SWAG training..."
python "$RUN_SCRIPT" \
    --dir="$OUTPUT_DIR" \
    --dataset=two_moons \
    --data_path="$DATA_PATH" \
    --model=TwoDMLP \
    --epochs=200 \
    --lr_init=0.1 \
    --wd=3e-4 \
    --swa \
    --swa_lr=0.05 \
    --swa_start=104 \
    --use_test \
    --cov_mat 

echo "Two moons training completed."
