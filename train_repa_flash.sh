cd ~/DiT
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=3,4,6,7 $CONDA_PREFIX/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=4 train_repa_flash.py \
  --model DiT-XL/2 \
  --data-path /data/ImageNet1k/train \
  --global-batch-size 64 \
  --use-repa \
  --proj-coeff 0.5 \
  --repa-input-size 224 \
  --repa-r 1 \
  --repa-r-strategy fixed \
  --repa-stop-step 200000 \
  --results-dir /data4/haksoo/trm_repa \
  --amp fp16 \
  --enc-type dinov2-vit-b \
  --trm-mode self_refine \
  --ckpt /data4/haksoo/trm_repa/003-DiT-XL-2/checkpoints/0160000.pt