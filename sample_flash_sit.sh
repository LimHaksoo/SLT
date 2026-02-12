#!/usr/bin/env bash
set -e

# Example usage:
#   bash sample_flash_sit.sh /path/to/checkpoint.pt

CKPT=${1:-""}
if [ -z "$CKPT" ]; then
  echo "Usage: bash sample_flash_sit.sh /path/to/checkpoint.pt"
  exit 1
fi

python sample_flash_sit.py \
  --ckpt "$CKPT" \
  --model DiT-XL/2 \
  --image-size 256 \
  --num-samples 64 \
  --batch-size 8 \
  --path-type Linear \
  --prediction velocity \
  --sampler ode \
  --ode-method Heun \
  --num-steps 50 \
  --cfg-scale 1.5 \
  --outdir samples_sit
