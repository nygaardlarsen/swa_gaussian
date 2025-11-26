 #!/bin/bash
# ========================================
# SWAG-Gaussian run script
# ========================================

set -e  # Stop on errors

# --- Paths ---
DATA_PATH="/work/SWAG/data/cifar"
MODEL_PATH="/work/SWAG/models/PreResNet56_cifar10_swag/swag-100.pt"
SAVE_PATH="/work/SWAG/models/PreResNet56_cifar10_swa_eval"
RUN_SCRIPT="/work/SWAG/experiments/uncertainty/uncertainty.py"

# Opret output directory, hvis det ikke findes
mkdir -p "$SAVE_PATH"

# --- Kør SWA eval ---
echo "Evaluating SWA model..."
python "$RUN_SCRIPT" \
    --file="$MODEL_PATH" \
    --save_path="$SAVE_PATH" \
    --dataset=CIFAR10 \
    --data_path="$DATA_PATH" \
    --model=PreResNet56 \
    --method=SWAG \
    --scale=0. \
    --cov_mat \
    --use_test \
    --use_diag \
    --N=1

echo "SWA eval completed."
 