
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# NOTE: This is a "flash/AMP/compile"-friendly sampling script based on:
#   - sample_ddp.py (DDP sampling for FID .npz)
#   - sample.py     (single-run sampling logic)
# It is intended to work with the "train_flash.py" setup and the
# weight-tied TRM-DiT models (e.g., model_dittrm_flash.py).
#
"""
Samples a large number of images from a pre-trained DiT (or TRM-DiT) model using DDP.
Saves .png files and a single .npz file usable for FID computation (ADM eval scripts).

Usage (2 GPUs):
  torchrun --nnodes=1 --nproc_per_node=2 sample_flash.py \
    --model DiT-XL/2 --image-size 256 --num-fid-samples 50000 \
    --per-proc-batch-size 32 --cfg-scale 1.5 --num-sampling-steps 250 \
    --ckpt /path/to/checkpoint.pt

Notes:
  - If --ckpt is omitted, this script expects the official DiT naming convention and uses `download.find_model`.
  - AMP (bf16/fp16) and torch.compile are optional speed knobs.
  - For TRM-DiT, you can enable/disable adaptive halting at sampling time via flags.
"""

from __future__ import annotations

import argparse
import math
import os
from time import time
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist

# Performance knobs (same spirit as train_flash.py):
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    # Encourage FlashAttention / mem-efficient SDPA kernels when available:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Model registry (flash variant if present):
try:
    from model_dittrm_flash import DiT_models  # type: ignore
except Exception:
    from model_dittrm import DiT_models  # type: ignore

from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


def create_npz_from_sample_folder(sample_dir: str, num: int = 50_000) -> str:
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def _set_trm_runtime_flags(model: torch.nn.Module, args: argparse.Namespace) -> None:
    """
    Adjust runtime-only TRM flags (no effect on checkpoint weights).
    This is safe for both baseline DiT and TRM-DiT models.
    """
    # Attributes exist only for TRM-DiT. We gate with hasattr to stay drop-in compatible.
    if hasattr(model, "adaptive_halt"):
        model.adaptive_halt = bool(args.adaptive_halt)
    if hasattr(model, "halt_eps") and args.halt_eps is not None:
        model.halt_eps = float(args.halt_eps)
    if hasattr(model, "min_steps") and args.min_steps is not None:
        model.min_steps = int(args.min_steps)


def main(args: argparse.Namespace) -> None:
    # TF32 control:
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    assert torch.cuda.is_available(), (
        "Sampling with DDP requires at least one GPU. "
        "Use torchrun --nproc_per_node=1 for single-GPU DDP."
    )
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)

    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    dist.barrier()

    # If auto-downloading, enforce canonical config:
    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # AMP config:
    amp_enabled = args.amp != "none"
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    ).to(device)

    # Load checkpoint (pretrained or training checkpoint):
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0 and (missing or unexpected):
        # Many training checkpoints are compatible; strict=False makes this robust for minor metadata differences.
        print(f"[load_state_dict] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    # Optional torch.compile:
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except TypeError:
            model = torch.compile(model)

    model.eval()  # important!

    # TRM runtime knobs:
    _set_trm_runtime_flags(model, args)

    diffusion = create_diffusion(str(args.num_sampling_steps))

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    if amp_dtype is not None:
        try:
            vae.to(dtype=amp_dtype)
        except Exception:
            pass

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale should be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-"
        f"cfg-{args.cfg_scale}-seed-{args.global_seed}-amp-{args.amp}"
    )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Determine sampling workload:
    n = args.per_proc_batch_size
    global_batch_size = n * world_size
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % world_size == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // world_size)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by per-proc batch size"
    iterations = int(samples_needed_this_gpu // n)

    # Progress bar on rank 0:
    pbar = range(iterations)
    pbar = tqdm(pbar, desc="Sampling") if rank == 0 else pbar

    # Wrap model calls with autocast to trigger FlashAttention kernels where possible:
    def sample_fn(x: torch.Tensor, t: torch.Tensor, **model_kwargs):
        if amp_enabled and amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model(x, t, **model_kwargs)
        return model(x, t, **model_kwargs)

    def sample_fn_cfg(x: torch.Tensor, t: torch.Tensor, **model_kwargs):
        if amp_enabled and amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model.forward_with_cfg(x, t, **model_kwargs)
        return model.forward_with_cfg(x, t, **model_kwargs)

    total_done = 0
    t0 = time()

    for _ in pbar:
        # Sample noise + labels:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.full((n,), args.num_classes, device=device, dtype=y.dtype)  # null class id
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            fn = sample_fn_cfg
        else:
            model_kwargs = dict(y=y)
            fn = sample_fn

        # Denoise:
        samples = diffusion.p_sample_loop(
            fn,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null-class samples

        # Decode to pixel space:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            decoded = vae.decode(samples / 0.18215).sample

        decoded = torch.clamp(127.5 * decoded + 128.0, 0, 255)
        decoded = decoded.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save as PNG:
        for i, sample in enumerate(decoded):
            index = i * world_size + rank + total_done
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total_done += global_batch_size

        # Optional speed print on rank 0:
        if rank == 0 and isinstance(pbar, tqdm) and (pbar.n % max(1, args.log_every_iters) == 0):
            dt = time() - t0
            ips = total_done / max(dt, 1e-9)
            pbar.set_postfix({"img/s": f"{ips:.2f}", "total": total_done})

    # Sync all ranks before .npz building:
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Use TF32 matmuls (fast on Ampere+).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a DiT checkpoint (default: auto-download a pretrained DiT-XL/2).")

    # Speed knobs:
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16",
                        help="AMP dtype for sampling. bf16 recommended on Ampere+.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for the sampling model (first call is slower).")
    parser.add_argument("--compile-mode", type=str, default="max-autotune",
                        help="torch.compile mode (default/reduce-overhead/max-autotune).")
    parser.add_argument("--log-every-iters", type=int, default=10,
                        help="Rank0 progress postfix update frequency (iterations).")

    # TRM runtime knobs (no effect for baseline DiT):
    parser.add_argument("--adaptive-halt", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable TRM adaptive halting at inference (TRM models only).")
    parser.add_argument("--halt-eps", type=float, default=None,
                        help="TRM halting threshold on mean|Δepsilon| (TRM models only).")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="TRM minimum steps before halting can trigger (TRM models only).")

    args = parser.parse_args()
    main(args)
