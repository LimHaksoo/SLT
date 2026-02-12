# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# sample_repa_flash_sit.py
# -----------------------------------------------------------------------------
# DDP sampling script intended for models trained with train_repa_flash_sit.py
# (i.e., SiT-style Transport objective + optional REPA projector weights).
#
# Key properties:
#   - Loads EMA weights if present (recommended for sampling) via --use-ema
#   - Uses strict=False when loading state dict (so extra keys like repa_projector don't break)
#   - Supports AMP + torch.compile + TF32/SDPA knobs (flash-friendly)
#   - Supports TRM runtime flags (adaptive halting) at inference time
#   - Generates PNGs and (optionally) packs the first `--num-fid-samples` into a single .npz
#
# Note: REPA affects training only; sampling does not require the teacher/projector.
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
# Performance knobs
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Prefer Flash / mem-efficient SDPA when available, but keep math as fallback.
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
from model_dittrm_flash import DiT_models  # type: ignore

from diffusers.models import AutoencoderKL

# SiT transport sampler:
from transport import create_transport, Sampler  # type: ignore


def create_npz_from_sample_folder(sample_dir: str, num: int = 50_000) -> str:
    """Build a single .npz file from a folder of .png samples."""
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
    """Adjust runtime-only TRM flags (no effect on checkpoint weights)."""
    if hasattr(model, "adaptive_halt"):
        model.adaptive_halt = bool(args.adaptive_halt)
    if hasattr(model, "halt_eps") and args.halt_eps is not None:
        model.halt_eps = float(args.halt_eps)
    if hasattr(model, "min_steps") and args.min_steps is not None:
        model.min_steps = int(args.min_steps)


def _load_training_checkpoint(path: str, *, use_ema: bool = True) -> Tuple[Dict[str, torch.Tensor], Optional[Any]]:
    """
    Load a checkpoint path. Supports:
      - training checkpoints saved as dict containing 'model'/'ema'
      - raw state_dict

    Returns:
      (state_dict, ckpt_args_obj_or_None)
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, dict) and (("model" in obj) or ("ema" in obj)):
        if use_ema and ("ema" in obj):
            sd = obj["ema"]
        elif "model" in obj:
            sd = obj["model"]
        else:
            sd = obj["ema"]
        ckpt_args = obj.get("args", None)
        return sd, ckpt_args

    return obj, None


def _infer_trm_kwargs_from_ckpt_args(args: argparse.Namespace, ckpt_args: Any) -> None:
    """Infer TRM architecture knobs from checkpoint args if user did not pass them."""
    if ckpt_args is None:
        return
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
    np.random.seed(seed)

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
    assert args.ckpt is not None, "--ckpt is required for SiT sampling."
    state_dict, ckpt_args_obj = _load_training_checkpoint(args.ckpt, use_ema=args.use_ema)

    # Infer TRM architecture knobs from ckpt args if user didn't override:
    _infer_trm_kwargs_from_ckpt_args(args, ckpt_args_obj)

    # Create model:
    latent_size = args.image_size // 8
    model_ctor = DiT_models[args.model]

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

    # learn_sigma is usually True in this codebase; we keep it configurable:
    ctor_kwargs["learn_sigma"] = bool(args.learn_sigma)

    try:
        model = model_ctor(**ctor_kwargs).to(device)
    except TypeError:
        # Vanilla DiT fallback (ignore TRM kwargs):
        ctor_kwargs = dict(input_size=latent_size, num_classes=args.num_classes, learn_sigma=bool(args.learn_sigma))
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

    # Transport + sampler:
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
    )
    sampler = Sampler(transport)
    if args.sampler == "ode":
        sample_fn = sampler.sample_ode(
            sampling_method=args.ode_method,
            num_steps=args.num_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=False,
        )
    elif args.sampler == "sde":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sde_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_steps,
        )
    else:
        raise ValueError(f"Unknown sampler={args.sampler}")

    # VAE:
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
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "")
    ema_tag = "ema" if args.use_ema else "model"
    folder_name = (
        f"{model_string_name}-{ckpt_string_name}-{ema_tag}-"
        f"size-{args.image_size}-vae-{args.vae}-cfg-{args.cfg_scale}-"
        f"sit-{args.path_type}-{args.prediction}-"
        f"{args.sampler}-{args.num_steps}-"
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
    assert total_samples % world_size == 0
    samples_needed_this_gpu = int(total_samples // world_size)
    assert samples_needed_this_gpu % n == 0
    iterations = int(samples_needed_this_gpu // n)

    # Wrap model calls with autocast:
    autocast_device = "cuda"
    def _model_pred(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Return prediction with shape == x (C channels), applying (optional) CFG.
        - If learn_sigma=True, we slice the first C channels.
        - Guidance is applied across ALL latent channels.
        """
        if t.dtype != torch.float32:
            t = t.float()

        if not using_cfg:
            if amp_enabled and amp_dtype is not None:
                with torch.autocast(device_type=autocast_device, dtype=amp_dtype):
                    out = model(x, t, y)
            else:
                out = model(x, t, y)
            if out.shape[1] == 2 * x.shape[1]:
                out = out[:, : x.shape[1]]
            return out.float()

        # CFG in one forward:
        y_null = torch.full_like(y, args.num_classes)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y, y_null], dim=0)
        if amp_enabled and amp_dtype is not None:
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype):
                out = model(x_in, t_in, y_in)
        else:
            out = model(x_in, t_in, y_in)
        if out.shape[1] == 2 * x.shape[1]:
            out = out[:, : x.shape[1]]
        out = out.float()
        cond, uncond = out.chunk(2, dim=0)
        return uncond + args.cfg_scale * (cond - uncond)

    total_done = 0
    t0 = time()

    pbar = range(iterations)
    pbar = tqdm(pbar, desc="Sampling") if rank == 0 else pbar

    for _ in pbar:
        # Initial noise:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device, dtype=torch.float32)

        # Labels:
        if args.class_label is None:
            y = torch.randint(0, args.num_classes, (n,), device=device)
        else:
            y = torch.full((n,), int(args.class_label), device=device, dtype=torch.long)

        # Sample latents:
        samples = sample_fn(z, _model_pred, y=y)

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

        if rank == 0 and isinstance(pbar, tqdm) and (pbar.n % max(1, args.log_every_iters) == 0):
            dt = time() - t0
            ips = min(total_done, total_samples) / max(dt, 1e-9)
            pbar.set_postfix({"img/s": f"{ips:.2f}", "done": min(total_done, total_samples)})

    dist.barrier()
    if rank == 0 and (not args.no_npz):
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
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--ckpt", type=str, required=True, help="Path to a training checkpoint (.pt).")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)

    # SiT transport config:
    parser.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "noise", "score"])
    parser.add_argument("--loss-weight", type=str, default=None, choices=["velocity", "likelihood"])
    parser.add_argument("--train-eps", type=float, default=None)
    parser.add_argument("--sample-eps", type=float, default=None)

    # Sampler config:
    parser.add_argument("--sampler", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument("--num-steps", type=int, default=50)

    # ODE options:
    parser.add_argument("--ode-method", type=str, default="Heun", choices=["Euler", "Heun"])
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)

    # SDE options:
    parser.add_argument("--sde-method", type=str, default="Euler", choices=["Euler"])
    parser.add_argument("--diffusion-form", type=str, default="SBDM", choices=["SBDM", "constant", "linear", "decreasing"])
    parser.add_argument("--diffusion-norm", type=float, default=1.0)
    parser.add_argument("--last-step", type=str, default=None, choices=["Mean", "Euler", "Tweedie"])
    parser.add_argument("--last-step-size", type=float, default=0.0)

    # Optional: fix class label (otherwise random):
    parser.add_argument("--class-label", type=int, default=None)

    # Speed knobs:
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="max-autotune")
    parser.add_argument("--log-every-iters", type=int, default=10)

    # TRM architecture knobs (optional; inferred from ckpt args if None):
    parser.add_argument("--shared-depth", type=int, default=None)
    parser.add_argument("--trm-mode", type=str, default=None, choices=["latent", "self_refine"])
    parser.add_argument("--H-cycles", type=int, default=None)
    parser.add_argument("--L-cycles", type=int, default=None)

    # TRM runtime knobs (no effect for baseline DiT):
    parser.add_argument("--adaptive-halt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--halt-eps", type=float, default=None)
    parser.add_argument("--min-steps", type=int, default=None)

    # learn_sigma:
    parser.add_argument("--learn-sigma", action="store_true", default=True)
    parser.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")

    # Debug:
    parser.add_argument("--print-load-keys", action="store_true")

    # Convenience:
    parser.add_argument("--no-npz", action="store_true", help="Do not pack PNGs into .npz at the end.")

    args = parser.parse_args()
    main(args)
