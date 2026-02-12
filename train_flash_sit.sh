cd ~/diffusion_1/DiT
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=0,2 $CONDA_PREFIX/bin/python -m torch.distributed.run \
  --nproc_per_node=2 train_flash_sit.py --model DiT-XL/2 --data-path /data/ImageNet1k/train \
  --global-batch-size 64 --amp bf16 \
  --path-type Linear --prediction velocity
