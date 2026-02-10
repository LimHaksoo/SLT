#!/bin/bash

cd ~/DiT
export PYTHONNOUSERSITE=1

# 1. 공통 경로 및 설정 변수화
GPU_ID=5
BASE_DIR="/data4/haksoo/trm_repa/003-DiT-XL-2/checkpoints"
CKPT="${BASE_DIR}/0160000.pt"

# 공통적으로 사용되는 옵션들
COMMON_ARGS="
  --nnodes=1 --nproc_per_node=1 --master_port=29501 sample_repa.py
  --ckpt $CKPT
  --model DiT-XL/2
  --image-size 256
  --num-fid-samples 64
  --per-proc-batch-size 16
  --cfg-scale 1.5
  --num-sampling-steps 250
  --amp fp16
"

# 2. 실행 로직을 함수로 정의
run_sampling() {
    local OUT_DIR_NAME=$1
    local EXTRA_ARGS=$2
    
    echo "Running sampling -> Output: ${OUT_DIR_NAME}"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_PREFIX/bin/python -m torch.distributed.run \
      $COMMON_ARGS \
      --sample-dir "${BASE_DIR}/${OUT_DIR_NAME}" \
      $EXTRA_ARGS
}

# --- 실행 명령 ---

# Case 1: 기본 실행
run_sampling "samples" ""

# Case 2: EMA 끄기 (samples_raw)
run_sampling "samples_raw" "--use-ema False"

# Case 3: Adaptive Halting (samples_halt)
run_sampling "samples_halt" "--adaptive-halt --halt-eps 1e-3 --min-steps 2"