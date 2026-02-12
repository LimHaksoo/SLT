# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SiT-style sampling script adapted for DiT/TRM codebase (latent-space ImageNet).

"""
Sample images from a trained SiT-style Transport model (DiT backbone).

This script mirrors the DiT "sample.py / sample_ddp.py" style:
  - loads a DiT(-TRM) checkpoint (prefers EMA weights)
  - samples latents by solving the probability flow ODE (Euler/Heun)
  - decodes latents with SD VAE and writes PNGs

Usage examples:

Single GPU, ODE-Heun:
  python sample_flash_sit.py --ckpt results/.../checkpoints/0040000.pt \
      --model DiT-XL/2 --image-size 256 --num-samples 64 --batch-size 8 \
      --path-type Linear --prediction velocity --ode-method Heun --num-steps 50 \
      --cfg-scale 1.5 --outdir samples_sit

Multi-GPU (DDP) to generate many images:
  torchrun --nproc_per_node=8 sample_flash_sit.py --ckpt ... --num-samples 50000 --batch-size 16 --outdir samples_sit_50k
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL

from model_dittrm_flash import DiT_models
from transport import create_transport, Sampler


def _setup_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


@torch.no_grad()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    """
    Decode DiT latents to images in [0,1], float32.
    DiT training uses latent scaling factor 0.18215.
    """
    # VAE expects fp16/bf16 is okay; output we'll keep fp32.
    imgs = vae.decode(latents / 0.18215).sample
    imgs = (imgs + 1) / 2
    return imgs.clamp(0, 1).float()


def save_png_batch(imgs01: torch.Tensor, outdir: Path, start_idx: int):
    """
    imgs01: [B,3,H,W] in [0,1]
    """
    imgs = (imgs01 * 255.0).round().clamp(0, 255).to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # BCHW -> BHWC
    for i, arr in enumerate(imgs):
        Image.fromarray(arr).save(outdir / f"{start_idx + i:06d}.png")


def main(args):
    # Speed niceties:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    rank, world_size, device = _setup_distributed()

    # Seeding per-rank:
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Output:
    outdir = Path(args.outdir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    # Load VAE:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # Build model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=args.learn_sigma,
    ).to(device)
    model.eval()

    # Load checkpoint (prefer EMA):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = None
    if args.use_ema and isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        if missing:
            print(f"[warn] Missing keys when loading ckpt: {len(missing)}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading ckpt: {len(unexpected)}")

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

    # AMP for model forward (keep latents in fp32 for solver stability):
    amp_enabled = args.amp in ["fp16", "bf16"]
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    def model_pred(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Return prediction with shape == x (C channels), applying (optional) CFG.

        Notes:
          - We apply guidance across ALL latent channels (not just the first 3).
          - If learn_sigma=True, we slice the prediction head to first C channels.
        """
        # Ensure t is float32 for embedding stability:
        if t.dtype != torch.float32:
            t = t.float()

        if args.cfg_scale <= 1.0:
            with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_enabled):
                out = model(x, t, y)
            if out.shape[1] == 2 * x.shape[1]:
                out = out[:, : x.shape[1]]
            return out.float()

        # Classifier-free guidance in a single forward pass:
        y_null = torch.full_like(y, args.num_classes)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([y, y_null], dim=0)
        with torch.autocast(device_type=autocast_device, dtype=amp_dtype, enabled=amp_enabled):
            out = model(x_in, t_in, y_in)
        if out.shape[1] == 2 * x.shape[1]:
            out = out[:, : x.shape[1]]
        out = out.float()
        cond, uncond = out.chunk(2, dim=0)
        return uncond + args.cfg_scale * (cond - uncond)

    # How many samples this rank should do:
    total = int(args.num_samples)
    per_rank = int(math.ceil(total / world_size))
    start = rank * per_rank
    end = min(start + per_rank, total)

    if start >= end:
        # This rank has no work.
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    # Sampling loop:
    cur = start
    while cur < end:
        b = min(args.batch_size, end - cur)

        # Labels:
        if args.class_label is None:
            y = torch.randint(low=0, high=args.num_classes, size=(b,), device=device)
        else:
            y = torch.full((b,), int(args.class_label), device=device, dtype=torch.long)

        # Initial noise:
        x = torch.randn(b, 4, latent_size, latent_size, device=device, dtype=torch.float32)

        # Sample latents:
        x = sample_fn(x, model_pred, y=y)

        # Decode and save:
        imgs = decode_latents(vae, x)
        save_png_batch(imgs, outdir, cur)

        cur += b

    # Optional: save a grid on rank0 for quick sanity check (first batch only).
    if rank == 0 and args.save_grid:
        # Load a few images back (avoid re-decode):
        files = sorted(outdir.glob("*.png"))[: min(args.grid_n, total)]
        if files:
            imgs = [torch.from_numpy(np.array(Image.open(f)).transpose(2, 0, 1)) for f in files]
            imgs = torch.stack(imgs).float() / 255.0
            save_image(imgs, outdir / "grid.png", nrow=int(math.sqrt(len(imgs))))

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Checkpoint / model:
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-XL/2", choices=list(DiT_models.keys()))
    parser.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--learn-sigma", action="store_true", default=True)
    parser.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")

    # Sampling settings:
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="samples_sit")
    parser.add_argument("--class-label", type=int, default=None, help="If set, generate only this class id.")

    # Transport config:
    parser.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "noise", "score"])
    parser.add_argument("--loss-weight", type=str, default=None, choices=["velocity", "likelihood"])
    parser.add_argument("--train-eps", type=float, default=None)
    parser.add_argument("--sample-eps", type=float, default=None)

    # ODE / SDE choice:
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

    # Guidance / AMP / VAE:
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--amp", type=str, default="bf16", choices=["none", "fp16", "bf16"])
    parser.add_argument("--vae", type=str, default="ema", choices=["ema", "mse"])

    # Convenience:
    parser.add_argument("--save-grid", action="store_true")
    parser.add_argument("--grid-n", type=int, default=64)

    args = parser.parse_args()
    main(args)
