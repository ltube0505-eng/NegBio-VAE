#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python dis_main.py \
  --config_path configs/hnbconfig.yaml \
  --dataset MNIST \
  --model_type hnb \
  --reparam_type gamma \
  --kl mc \
  --tau 1.0 \
  --hnb_groups_per_scale 4,4 \
  --hnb_channels_per_scale 64,32 \
  --hnb_latent_channels_per_scale 16,16 \
  --bsize 64 \
  --max_epochs 300
