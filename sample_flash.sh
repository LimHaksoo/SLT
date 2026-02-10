cd ~/diffusion_1/DiT
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=0,2 $CONDA_PREFIX/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=2 --master_port=29501 sample_flash.py \
  --model DiT-XL/2 \
  --image-size 256 \
  --num-fid-samples 64 \
  --per-proc-batch-size 32 \
  --cfg-scale 1.5 \
  --num-sampling-steps 250 \
  --ckpt /home/myeongjin/diffusion_1/DiT/results/003-DiT-XL-2/checkpoints/0020000.pt \
  --amp bf16