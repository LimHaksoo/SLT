
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# sample_repa.py
# -----------------------------------------------------------------------------
# DDP sampling script intended for models trained with train_repa_flash.py
# (i.e., checkpoints may include keys: {"model","ema","opt","args","train_steps"} and
# may contain additional REPA projector weights).
#
# This script is intentionally robust:
#   - Loads EMA weights if present (recommended for sampling) via --use-ema
#   - Uses strict=False when loading state dict (so extra keys like repa_projector don't break)
#   - Supports AMP + torch.compile + TF32/SDPA knobs (flash-friendly like sample_flash.py)
#   - Supports TRM runtime flags (adaptive halting) at inference time
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import math
import os
from time import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Performance knobs (similar spirit as train_flash.py / sample_flash.py)
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Prefer Flash / mem-efficient SDPA when available, but KEEP math as fallback to
# avoid "Invalid backend" crashes in edge cases.
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
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
    Safe for both baseline DiT and TRM-DiT models.
    """
    if hasattr(model, "adaptive_halt"):
        model.adaptive_halt = bool(args.adaptive_halt)
    if hasattr(model, "halt_eps") and args.halt_eps is not None:
        model.halt_eps = float(args.halt_eps)
    if hasattr(model, "min_steps") and args.min_steps is not None:
        model.min_steps = int(args.min_steps)


def _load_training_checkpoint(path: str, *, use_ema: bool = True) -> Tuple[Dict[str, torch.Tensor], Optional[Any]]:
    """
    Load a checkpoint path. Supports:
      - official pretrained weights (state_dict only)
      - training checkpoints saved as dict containing 'model'/'ema'

    Returns:
      (state_dict, ckpt_args_obj_or_None)
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    # Training checkpoint style: {"model": ..., "ema": ..., ...}
    if isinstance(obj, dict) and (("model" in obj) or ("ema" in obj)):
        if use_ema and ("ema" in obj):
            sd = obj["ema"]
        elif "model" in obj:
            sd = obj["model"]
        else:
            sd = obj["ema"]
        ckpt_args = obj.get("args", None)
        return sd, ckpt_args

    # Pretrained style: raw state_dict
    return obj, None


def _infer_trm_kwargs_from_ckpt_args(args: argparse.Namespace, ckpt_args: Any) -> None:
    """
    If user didn't explicitly pass TRM architecture knobs, but the checkpoint
    stores them (train_repa_flash.py saves args), then fill them in so the
    architecture matches.
    """
    if ckpt_args is None:
        return

    # argparse defaults are None (sentinel) for these in this script.
    for name in ["shared_depth", "trm_mode", "H_cycles", "L_cycles"]:
        if getattr(args, name) is None and hasattr(ckpt_args, name):
            setattr(args, name, getattr(ckpt_args, name))


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

    # AMP config:
    amp_enabled = args.amp != "none"
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    # Resolve checkpoint:
    ckpt_args_obj = None
    if args.ckpt is None:
        # Auto-download path (official DiT). In this mode, we assume canonical settings.
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 is available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        ckpt_path = f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
    else:
        state_dict, ckpt_args_obj = _load_training_checkpoint(args.ckpt, use_ema=args.use_ema)

    # If checkpoint had saved args, use them for TRM architecture unless overridden:
    _infer_trm_kwargs_from_ckpt_args(args, ckpt_args_obj)

    # Create model:
    latent_size = args.image_size // 8
    model_ctor = DiT_models[args.model]

    # Try to pass TRM knobs if available; fall back if constructor doesn't accept them.
    ctor_kwargs = dict(
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    # Only include if user/ckpt provided (None means "don't pass"):
    if args.shared_depth is not None:
        ctor_kwargs["shared_depth"] = int(args.shared_depth)
    if args.trm_mode is not None:
        ctor_kwargs["trm_mode"] = str(args.trm_mode)
    if args.H_cycles is not None:
        ctor_kwargs["H_cycles"] = int(args.H_cycles)
    if args.L_cycles is not None:
        ctor_kwargs["L_cycles"] = int(args.L_cycles)

    try:
        model = model_ctor(**ctor_kwargs).to(device)
    except TypeError:
        # For vanilla DiT constructors (if any), ignore TRM kwargs:
        ctor_kwargs = dict(input_size=latent_size, num_classes=args.num_classes)
        model = model_ctor(**ctor_kwargs).to(device)

    # Load weights (strict=False to tolerate extra keys like repa_projector):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0 and (missing or unexpected):
        print(f"[load_state_dict] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if args.print_load_keys:
            if missing:
                print("  missing (first 20):", missing[:20])
            if unexpected:
                print("  unexpected (first 20):", unexpected[:20])

    # Optional torch.compile:
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except TypeError:
            model = torch.compile(model)

    model.eval()

    # TRM runtime knobs (early exit etc):
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
    ema_tag = "ema" if (args.ckpt and args.use_ema) else "model"
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-{ema_tag}-"
        f"size-{args.image_size}-vae-{args.vae}-cfg-{args.cfg_scale}-"
        f"seed-{args.global_seed}-amp-{args.amp}"
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

    # Wrap model calls with autocast:
    def _model_forward(x: torch.Tensor, t: torch.Tensor, **model_kwargs):
        if amp_enabled and amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return model(x, t, **model_kwargs)
        return model(x, t, **model_kwargs)

    def _model_forward_cfg(x: torch.Tensor, t: torch.Tensor, **model_kwargs):
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
            fn = _model_forward_cfg
        else:
            model_kwargs = dict(y=y)
            fn = _model_forward

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
            if index >= total_samples:
                break
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total_done += global_batch_size

        # Optional speed print on rank 0:
        if rank == 0 and isinstance(pbar, tqdm) and (pbar.n % max(1, args.log_every_iters) == 0):
            dt = time() - t0
            ips = min(total_done, total_samples) / max(dt, 1e-9)
            pbar.set_postfix({"img/s": f"{ips:.2f}", "done": min(total_done, total_samples)})

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
    parser.add_argument("--tf32", type=bool, default=True,
                        help="Use TF32 matmuls (fast on Ampere+).")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a training checkpoint (.pt). If omitted, auto-download a pretrained DiT-XL/2.")
    parser.add_argument("--use-ema", type=bool, default=True,
                        help="If checkpoint contains EMA weights, use them for sampling (recommended).")

    # Speed knobs:
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16",
                        help="AMP dtype for sampling. bf16 recommended on Ampere+.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for the sampling model (first call is slower).")
    parser.add_argument("--compile-mode", type=str, default="max-autotune",
                        help="torch.compile mode (default/reduce-overhead/max-autotune).")
    parser.add_argument("--log-every-iters", type=int, default=10,
                        help="Rank0 progress postfix update frequency (iterations).")

    # TRM architecture knobs (optional). If omitted and checkpoint stores args, we infer them.
    parser.add_argument("--shared-depth", type=int, default=None,
                        help="(TRM) number of unique blocks instantiated; if None, infer from ckpt args if available.")
    parser.add_argument("--trm-mode", type=str, choices=["latent", "self_refine"], default=None,
                        help="(TRM) inference mode; if None, infer from ckpt args if available.")
    parser.add_argument("--H-cycles", type=int, default=None,
                        help="(TRM self_refine) outer cycles; if None, infer from ckpt args if available.")
    parser.add_argument("--L-cycles", type=int, default=None,
                        help="(TRM self_refine) inner cycles; if None, infer from ckpt args if available.")

    # TRM runtime knobs (no effect for baseline DiT):
    parser.add_argument("--adaptive-halt", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable TRM adaptive halting at inference (TRM models only).")
    parser.add_argument("--halt-eps", type=float, default=None,
                        help="TRM halting threshold on mean|Δepsilon| (TRM models only).")
    parser.add_argument("--min-steps", type=int, default=None,
                        help="TRM minimum steps before halting can trigger (TRM models only).")

    # Debug:
    parser.add_argument("--print-load-keys", action="store_true",
                        help="Print missing/unexpected keys (first 20) when loading checkpoints.")

    args = parser.parse_args()
    main(args)
