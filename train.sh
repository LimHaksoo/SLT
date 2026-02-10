cd ~/diffusion_1/DiT
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=2,3 $CONDA_PREFIX/bin/python -m torch.distributed.run \
  --nproc_per_node=2 train.py --model DiT-XL/2 --data-path /data1/dataset/imagenet2012/train --global-batch-size 64
