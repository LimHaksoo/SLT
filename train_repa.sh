cd ~/DiT
export PYTHONNOUSERSITE=1

CUDA_VISIBLE_DEVICES=4 $CONDA_PREFIX/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc_per_node=1 train_repa.py \
  --model DiT-XL/2 \
  --data-path /data/ImageNet1k/train \
  --global-batch-size 8 \
  --use-repa \
  --proj-coeff 0.5 \
  --repa-input-size 224 \
  --repa-r 1 \
  --repa-r-strategy fixed \
  --repa-stop-step 50000 \
  --results-dir /data4/haksoo/trm_repa