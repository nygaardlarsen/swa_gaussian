#!/bin/bash


BASE_OUTPUT_DIR="/work/SWAG/models/two_moons"
BASE_VIS_DIR="/work/SWAG/visualization"
TRAIN_SCRIPT="/work/SWAG/experiments/train/run_swag.py"
PLOT_SCRIPT="/work/SWAG/visualization/decision_boundaries_n_models.py"
DATA_PATH="/work/SWAG/data/two_moons"

NOISE_LEVELS=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7)

for noise in "${NOISE_LEVELS[@]}"
do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/noise_${noise}"
    MODEL_PATH="$OUTPUT_DIR/swag-200.pt"
    VIS_DIR="$BASE_VIS_DIR/noise_${noise}"
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$VIS_DIR"


    python "$TRAIN_SCRIPT" \
        --data_path="$DATA_PATH" \
        --dir="$OUTPUT_DIR" \
        --dataset=two_moons \
        --model=TwoDMLP \
        --n_data=500 \
        --noise=$noise \
        --epochs=200 \
        --lr_init=0.1 \
        --wd=3e-4 \
        --swa \
        --swa_start=104 \
        --swa_lr=0.05 \
        --cov_mat \
        --use_test

    python "$PLOT_SCRIPT" \
        --model_path="$MODEL_PATH" \
        --out_path="$VIS_DIR/two_moons_decision_boundary.png" \
        --n_bins=10 \
        --n_data=500 \
        --noise=$noise \
        --n_models=30
done