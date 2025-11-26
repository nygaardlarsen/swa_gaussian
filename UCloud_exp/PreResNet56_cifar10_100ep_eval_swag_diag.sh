  #!/bin/bash
# ========================================
# SWAG-Gaussian run script
# ========================================

set -e  # Stop on errors

# --- Paths ---
DATA_PATH="/work/SWAG/data/cifar"
MODEL_PATH="/work/SWAG/models/PreResNet56_cifar10_swag/swag-100.pt"
SAVE_PATH="/work/SWAG/models/PreResNet56_cifar10_swag_diag_eval"
RUN_SCRIPT="/work/SWAG/experiments/uncertainty/uncertainty.py"

# Opret output directory, hvis det ikke findes
mkdir -p "$SAVE_PATH"

# --- Kør SWAG-Diag eval ---
echo "Evaluating SWAG-Diag model..."
python "$RUN_SCRIPT" \
    --file="$MODEL_PATH" \
    --save_path="$SAVE_PATH" \
    --dataset=CIFAR10 \
    --data_path="$DATA_PATH" \
    --model=PreResNet56 \
    --method=SWAG \
    --use_test \
    --use_diag \
    --cov_mat

echo "SWAG-DIAG completed."
 