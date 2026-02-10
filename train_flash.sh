cd ~/DiT
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=4,5,6,7 $CONDA_PREFIX/bin/python -m torch.distributed.run \
  --nproc_per_node=4 train_flash.py --model DiT-XL/2 --data-path /data/ImageNet1k/train \
  --global-batch-size 256 --amp fp16 --results-dir /data4/haksoo \
  --ckpt /home/haksoo/DiT/results/003-DiT-XL-2/checkpoints/0020000.pt

