#!/bin/bash

cd ~/DiT
export PYTHONNOUSERSITE=1

# 1. 공통 경로 및 설정 변수화
GPU_ID=5
BASE_DIR="/data4/haksoo/trm_repa/008-DiT-XL-2/checkpoints"

# 순회할 체크포인트 번호들 (공백으로 구분)
CKPT_STEPS=("0460000")

# 2. 실행 로직을 함수로 정의
# 인자: $1=체크포인트 경로, $2=출력 폴더 접미사, $3=추가 인자
run_sampling() {
    local CKPT_PATH=$1
    local SUFFIX=$2
    local EXTRA_ARGS=$3
    
    # 체크포인트 파일명에서 숫자 추출 (예: 0340000.pt -> 0340000)
    local CKPT_NAME=$(basename "$CKPT_PATH" .pt)
    local OUT_DIR="${BASE_DIR}/${CKPT_NAME}_${SUFFIX}"

    echo "--------------------------------------------------"
    echo "Running sampling for CKPT: ${CKPT_NAME} -> Output: ${OUT_DIR}"
    echo "--------------------------------------------------"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_PREFIX/bin/python -m torch.distributed.run \
      --nnodes=1 --nproc_per_node=1 --master_port=29501 sample_repa.py \
      --ckpt "$CKPT_PATH" \
      --model DiT-XL/2 \
      --image-size 256 \
      --num-fid-samples 64 \
      --per-proc-batch-size 16 \
      --cfg-scale 4.0 \
      --num-sampling-steps 250 \
      --amp fp16 \
      --sample-dir "$OUT_DIR" \
      $EXTRA_ARGS
}

# --- 실행 명령 (반복문) ---

for STEP in "${CKPT_STEPS[@]}"
do
    CURRENT_CKPT="${BASE_DIR}/${STEP}.pt"
    
    # 파일 존재 여부 확인 (혹시 모를 에러 방지)
    if [ -f "$CURRENT_CKPT" ]; then
        # Case 1: 기본 실행 (samples)
        run_sampling "$CURRENT_CKPT" "samples" ""

        # Case 2: EMA 끄기 (samples_raw)
        run_sampling "$CURRENT_CKPT" "samples_raw" "--use-ema False"

        # Case 3: Adaptive Halting (samples_halt)
        run_sampling "$CURRENT_CKPT" "samples_halt" "--adaptive-halt --halt-eps 1e-3 --min-steps 2"
    else
        echo "Warning: ${CURRENT_CKPT} not found, skipping."
    fi
done

# 2. 실행 로직을 함수로 정의
# 인자: $1=체크포인트 경로, $2=출력 폴더 접미사, $3=추가 인자
run_sampling() {
    local CKPT_PATH=$1
    local SUFFIX=$2
    local EXTRA_ARGS=$3
    
    # 체크포인트 파일명에서 숫자 추출 (예: 0340000.pt -> 0340000)
    local CKPT_NAME=$(basename "$CKPT_PATH" .pt)
    local OUT_DIR="${BASE_DIR}/${CKPT_NAME}_${SUFFIX}"

    echo "--------------------------------------------------"
    echo "Running sampling for CKPT: ${CKPT_NAME} -> Output: ${OUT_DIR}"
    echo "--------------------------------------------------"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $CONDA_PREFIX/bin/python -m torch.distributed.run \
      --nnodes=1 --nproc_per_node=1 --master_port=29501 sample_repa.py \
      --ckpt "$CKPT_PATH" \
      --model DiT-XL/2 \
      --image-size 256 \
      --num-fid-samples 64 \
      --per-proc-batch-size 16 \
      --cfg-scale 1.5 \
      --num-sampling-steps 250 \
      --amp fp16 \
      --sample-dir "$OUT_DIR" \
      $EXTRA_ARGS
}

# --- 실행 명령 (반복문) ---

for STEP in "${CKPT_STEPS[@]}"
do
    CURRENT_CKPT="${BASE_DIR}/${STEP}.pt"
    
    # 파일 존재 여부 확인 (혹시 모를 에러 방지)
    if [ -f "$CURRENT_CKPT" ]; then
        # Case 1: 기본 실행 (samples)
        run_sampling "$CURRENT_CKPT" "samples" ""

        # Case 2: EMA 끄기 (samples_raw)
        run_sampling "$CURRENT_CKPT" "samples_raw" "--use-ema False"

        # Case 3: Adaptive Halting (samples_halt)
        run_sampling "$CURRENT_CKPT" "samples_halt" "--adaptive-halt --halt-eps 1e-3 --min-steps 2"
    else
        echo "Warning: ${CURRENT_CKPT} not found, skipping."
    fi
done