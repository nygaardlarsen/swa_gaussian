#!/bin/bash
set -e

cd swa_gaussian

DATA_PATH="/work/data/cifar"
OUTPUT_DIR="/work/models/PreResNet164_cifar10_swa"

mkdir -p $OUTPUT_DIR

python experiments/train/run_swag.py \
    --dir=$OUTPUT_DIR \
    --dataset=CIFAR10 \
    --data_path=$DATA_PATH \
    --model=PreResNet164 \
    --epochs=100 \
    --lr_init=0.1 \
    --wd=3e-4 \
    --swa \
    --swa_start=50 \
    --swa_lr=0.01 \
    --cov_mat \
    --use_test
