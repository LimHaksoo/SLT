# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a minimal training script for DiT using PyTorch DDP,
# **patched to optionally add REPA (REPresentation Alignment) loss** using a frozen
# pretrained visual encoder (default: DINOv2 ViT-B/14 via torch.hub).
#
# It remains compatible with the original DiT CLI and directory layout, and keeps
# diffusion/vae behavior unchanged. The patch is designed to be "minimally invasive":
# - main diffusion loss still comes from diffusion.training_losses(model, x, t, model_kwargs)
# - REPA loss is computed as an *extra regularizer* using a forward-hook to grab a
#   student token tensor at a chosen recurrent/block step (default r=1).
#
# Notes:
# - This script assumes your DiT model exposes `core_blocks` (as in model_dittrm.py).
#   If you use the original DiT, you can adapt the hook target to `blocks`.
# - Teacher encoder is frozen; gradients flow only into the student and the projector.

"""
A minimal training script for DiT using PyTorch DDP (with optional REPA regularization).
"""

import argparse
import logging
import math
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from model_dittrm import DiT_models
# from models import DiT_models

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                    REPA Utils                                 #
#################################################################################

_DINO_HUB_MAP = {
    "dinov2-vit-s": "dinov2_vits14",
    "dinov2-vit-b": "dinov2_vitb14",
    "dinov2-vit-l": "dinov2_vitl14",
    "dinov2-vit-g": "dinov2_vitg14",
    "dinov2-vit-s-reg": "dinov2_vits14_reg",
    "dinov2-vit-b-reg": "dinov2_vitb14_reg",
    "dinov2-vit-l-reg": "dinov2_vitl14_reg",
    "dinov2-vit-g-reg": "dinov2_vitg14_reg",
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def repa_lambda(step: int, *, base: float, warmup: int, stop: int, decay: int) -> float:
    """
    Piecewise schedule for REPA coefficient:
      - linear warmup for `warmup` steps up to `base`
      - constant `base` until `stop`
      - optional linear decay to 0 over `decay` steps
      - 0 afterwards
    """
    if base <= 0:
        return 0.0
    if warmup > 0 and step < warmup:
        return base * float(step) / float(max(1, warmup))
    if stop > 0 and step >= stop:
        if decay > 0 and step < stop + decay:
            # linear decay
            return base * (1.0 - float(step - stop) / float(decay))
        return 0.0
    return base


def preprocess_for_dino(x_img_m11: torch.Tensor, *, input_size: int) -> torch.Tensor:
    """
    x_img_m11: image tensor in [-1, 1], shape (B, 3, H, W)
    Returns normalized tensor for DINOv2 in ImageNet normalization.

    We resize to `input_size` to make patch grids match student tokens more often.
    For ImageNet-256 training, input_size=224 is a convenient default:
      - DINOv2 patch size is 14 => 224/14 = 16 patches per side (256 tokens).
    """
    x = (x_img_m11 + 1.0) * 0.5  # [-1,1] -> [0,1]
    if input_size is not None and (x.shape[-2] != input_size or x.shape[-1] != input_size):
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x


@torch.no_grad()
def dino_patch_tokens(
    teacher: torch.nn.Module,
    x_dino: torch.Tensor,
    *,
    encoder_depth: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns patch tokens of shape (B, N, D_t).

    If encoder_depth is provided and teacher exposes get_intermediate_layers(),
    we interpret encoder_depth as a 1-based layer index from the *start* and take
    that layer's patch tokens.

    Fallback: use forward_features()["x_norm_patchtokens"] (final layer, normalized).
    """
    if encoder_depth is not None and hasattr(teacher, "get_intermediate_layers"):
        # DINOv2 get_intermediate_layers accepts either:
        # - int: take n last layers
        # - list: take explicit layer indices (0-based)
        idx0 = max(0, int(encoder_depth) - 1)

        # Best-effort clamp to teacher depth if discoverable:
        try:
            if hasattr(teacher, "blocks"):
                # DINOv2 may store blocks as either a flat ModuleList or chunked blocks.
                if bool(getattr(teacher, "chunked_blocks", False)):
                    # See DINOv2 implementation: total_block_len = len(self.blocks[-1])
                    total_depth = len(teacher.blocks[-1])
                else:
                    total_depth = len(teacher.blocks)
                if total_depth > 0:
                    idx0 = min(idx0, total_depth - 1)
        except Exception:
            pass

        try:
            outs = teacher.get_intermediate_layers(
                x_dino,
                n=[idx0],
                reshape=False,
                return_class_token=False,
                norm=True,
            )
            # outs is a tuple
            return outs[0]  # (B, N, D)
        except Exception:
            # If anything goes wrong (API mismatch / out-of-range), fall back to final patch tokens.
            pass

    feats = teacher.forward_features(x_dino)
    # DINOv2 forward_features returns a dict with "x_norm_patchtokens"
    return feats["x_norm_patchtokens"]


def _reshape_tokens_to_map(tokens: torch.Tensor) -> Optional[torch.Tensor]:
    """
    tokens: (B, N, D) -> (B, D, H, W) if N is a square; else None
    """
    b, n, d = tokens.shape
    s = int(math.isqrt(n))
    if s * s != n:
        return None
    return tokens.transpose(1, 2).reshape(b, d, s, s)


def match_token_grid(teacher_tokens: torch.Tensor, student_tokens: torch.Tensor) -> torch.Tensor:
    """
    Ensure teacher token grid matches student token grid by interpolation if needed.
    Returns teacher_tokens_matched: (B, N_s, D_t)
    """
    if teacher_tokens.shape[1] == student_tokens.shape[1]:
        return teacher_tokens

    t_map = _reshape_tokens_to_map(teacher_tokens)
    s_map = _reshape_tokens_to_map(student_tokens)
    if t_map is None or s_map is None:
        # Fallback: global average teacher, broadcast to student token count
        t_global = teacher_tokens.mean(dim=1, keepdim=True)  # (B, 1, D_t)
        return t_global.expand(-1, student_tokens.shape[1], -1).contiguous()

    _, _, hs, ws = s_map.shape
    t_map_rs = F.interpolate(t_map, size=(hs, ws), mode="bilinear", align_corners=False)
    t_tokens_rs = t_map_rs.flatten(2).transpose(1, 2).contiguous()  # (B, N_s, D_t)
    return t_tokens_rs


def repa_cosine_loss(student_proj: torch.Tensor, teacher_tok: torch.Tensor) -> torch.Tensor:
    """
    Tokenwise cosine distance averaged over batch and tokens.
    Inputs:
      student_proj: (B, N, D_t)
      teacher_tok:  (B, N, D_t)
    """
    s = F.normalize(student_proj, dim=-1)
    t = F.normalize(teacher_tok, dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def attach_repa_projector(student: torch.nn.Module, teacher_dim: int, *, mlp: bool = False) -> None:
    """
    Create and attach a small projector to the student model as `student.repa_projector`.
    This makes it part of the student's state_dict and optimizer automatically.
    """
    # infer student token dim from its positional embedding
    if not hasattr(student, "pos_embed"):
        raise AttributeError("Student model must have .pos_embed to infer hidden size for REPA projector.")
    d_student = int(student.pos_embed.shape[-1])

    if mlp:
        proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_student),
            torch.nn.Linear(d_student, teacher_dim),
            torch.nn.GELU(),
            torch.nn.Linear(teacher_dim, teacher_dim),
        )
    else:
        proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_student),
            torch.nn.Linear(d_student, teacher_dim),
        )
    student.repa_projector = proj


def setup_repa_hooks(student: torch.nn.Module) -> Tuple[Dict[str, object], list]:
    """
    Register forward hooks on all recurrent blocks so we can capture the token tensor
    at a chosen block-call index r (1-based) inside student forward.

    Returns:
      state dict and list of hook handles (keep them alive!)
    """
    state: Dict[str, object] = {"count": 0, "target": 1, "z": None}

    def _hook(_module, _inputs, output):
        # output is tokens (B, N, D)
        state["count"] = int(state["count"]) + 1
        if int(state["count"]) == int(state["target"]):
            state["z"] = output
        return None

    handles = []
    if hasattr(student, "core_blocks"):
        blocks = list(student.core_blocks)
    elif hasattr(student, "blocks"):
        blocks = list(student.blocks)  # original DiT
    else:
        raise AttributeError("Student model must have `.core_blocks` (TRM-DiT) or `.blocks` (vanilla DiT).")

    for blk in blocks:
        handles.append(blk.register_forward_hook(_hook))
    return state, handles


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model (with optional REPA regularization).
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model_base = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        # --- TRM knobs (optional; ignored by vanilla DiT) ---
        shared_depth=args.shared_depth,
        trm_mode=args.trm_mode,
        H_cycles=args.H_cycles,
        L_cycles=args.L_cycles,
        adaptive_halt=False,  # don't early-stop during training
    )

    # Optional REPA teacher + projector:
    teacher = None
    repa_state = None
    repa_hook_handles = None
    if args.use_repa:
        assert args.enc_type in _DINO_HUB_MAP, f"Unknown --enc-type {args.enc_type}. Supported: {list(_DINO_HUB_MAP)}"
        hub_name = _DINO_HUB_MAP[args.enc_type]
        logger.info(f"Loading REPA teacher via torch.hub: facebookresearch/dinov2::{hub_name}")
        # DINOv2 models are downloaded from torch.hub in the REPA repo. (see their README)
        teacher = torch.hub.load("facebookresearch/dinov2", hub_name)
        teacher.eval().to(device)
        requires_grad(teacher, False)

        # Infer teacher dim by running a tiny forward on rank 0 and broadcasting dims.
        # (We still load teacher on all ranks; this just avoids shape-guessing.)
        if rank == 0:
            dummy = torch.zeros(1, 3, args.repa_input_size, args.repa_input_size, device=device)
            # already "normalized enough" for shape inference
            tok = dino_patch_tokens(teacher, dummy, encoder_depth=args.encoder_depth)
            teacher_dim = int(tok.shape[-1])
        else:
            teacher_dim = 0
        teacher_dim_t = torch.tensor([teacher_dim], device=device, dtype=torch.int64)
        dist.broadcast(teacher_dim_t, src=0)
        teacher_dim = int(teacher_dim_t.item())

        attach_repa_projector(model_base, teacher_dim, mlp=args.repa_proj_mlp)
        logger.info(f"Attached REPA projector: student_dim={int(model_base.pos_embed.shape[-1])} -> teacher_dim={teacher_dim}")

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model_base).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)

    model = DDP(model_base.to(device), device_ids=[device])

    # Now that the DDP model exists, we can set up hooks on the wrapped module.
    if args.use_repa:
        repa_state, repa_hook_handles = setup_repa_hooks(model.module)
        logger.info("Registered REPA forward hooks on student blocks.")

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (DiT paper used AdamW betas=(0.9, 0.999) and constant lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    if teacher is not None:
        teacher.eval()

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0.0
    running_repa = 0.0
    running_diff = 0.0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Keep a copy of images for teacher features (REPA) before encoding to latents.
            x_img = x

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x_lat = vae.encode(x_img).latent_dist.sample().mul_(0.18215)

            t = torch.randint(0, diffusion.num_timesteps, (x_lat.shape[0],), device=device)
            model_kwargs = dict(y=y)

            # Reset REPA hook state for this forward:
            if args.use_repa and repa_state is not None:
                repa_state["count"] = 0
                repa_state["z"] = None
                # Choose which block-call index to align (1-based):
                if args.repa_r_strategy == "fixed":
                    target_r = args.repa_r
                else:
                    # random
                    if hasattr(model.module, "trm_mode") and str(model.module.trm_mode) == "self_refine":
                        max_calls = int(model.module.H_cycles) * int(model.module.L_cycles)
                    else:
                        max_calls = int(getattr(model.module, "max_depth_steps", args.repa_r))
                    target_r = int(torch.randint(1, max_calls + 1, (1,), device=device).item())
                repa_state["target"] = target_r

            # Diffusion loss (unchanged):
            loss_dict = diffusion.training_losses(model, x_lat, t, model_kwargs)
            loss_diff = loss_dict["loss"].mean()

            # Optional REPA regularizer:
            loss_repa = None
            lam = 0.0
            if args.use_repa and teacher is not None and repa_state is not None:
                lam = repa_lambda(
                    train_steps,
                    base=args.proj_coeff,
                    warmup=args.repa_warmup_steps,
                    stop=args.repa_stop_step,
                    decay=args.repa_decay_steps,
                )
                if lam > 0:
                    z = repa_state["z"]
                    if z is None:
                        # If this happens, the hook did not fire (wrong hook target).
                        raise RuntimeError("REPA hook did not capture student tokens. Check model has core_blocks/blocks and repa_r is valid.")
                    # Teacher tokens from clean image:
                    with torch.no_grad():
                        x_dino = preprocess_for_dino(x_img, input_size=args.repa_input_size)
                        t_tok = dino_patch_tokens(teacher, x_dino, encoder_depth=args.encoder_depth)
                    # Project student tokens to teacher dim:
                    s_proj = model.module.repa_projector(z)
                    # Match token grids if needed:
                    t_tok = match_token_grid(t_tok, s_proj)
                    if t_tok.shape[1] != s_proj.shape[1]:
                        # still mismatched; fallback to global
                        t_tok = t_tok.mean(dim=1, keepdim=True).expand(-1, s_proj.shape[1], -1).contiguous()
                    loss_repa = repa_cosine_loss(s_proj, t_tok)

            loss = loss_diff + (lam * loss_repa if loss_repa is not None else 0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += float(loss.item())
            running_diff += float(loss_diff.item())
            if loss_repa is not None:
                running_repa += float(loss_repa.item())
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_total = torch.tensor(running_loss / log_steps, device=device)
                avg_diff = torch.tensor(running_diff / log_steps, device=device)
                avg_repa = torch.tensor(running_repa / log_steps, device=device)
                dist.all_reduce(avg_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_diff, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_repa, op=dist.ReduceOp.SUM)
                avg_total = avg_total.item() / dist.get_world_size()
                avg_diff = avg_diff.item() / dist.get_world_size()
                avg_repa = avg_repa.item() / dist.get_world_size()

                if args.use_repa:
                    logger.info(
                        f"(step={train_steps:07d}) "
                        f"Loss: {avg_total:.4f} (diff={avg_diff:.4f}, repa={avg_repa:.4f}, λ={lam:.3g}) "
                        f"Steps/Sec: {steps_per_sec:.2f}"
                    )
                else:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_total:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0.0
                running_diff = 0.0
                running_repa = 0.0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    # ---------------- TRM-DiT knobs (safe defaults) ----------------
    parser.add_argument("--shared-depth", type=int, default=1, help="Number of unique blocks to instantiate and reuse.")
    parser.add_argument("--trm-mode", type=str, default="latent", choices=["latent", "self_refine"])
    parser.add_argument("--H-cycles", type=int, default=3, help="Self-refine outer cycles (only when trm-mode=self_refine).")
    parser.add_argument("--L-cycles", type=int, default=4, help="Self-refine inner cycles per outer cycle.")

    # ---------------- REPA knobs ----------------
    parser.add_argument("--use-repa", action="store_true", help="Enable REPA loss using a frozen teacher encoder.")
    parser.add_argument("--enc-type", type=str, default="dinov2-vit-b", choices=list(_DINO_HUB_MAP.keys()))
    parser.add_argument("--encoder-depth", type=int, default=8, help="Teacher layer index (1-based) to take patch tokens from (if supported).")
    parser.add_argument("--proj-coeff", type=float, default=0.5, help="Base coefficient (λ) for REPA loss. 0 disables.")
    parser.add_argument("--repa-proj-mlp", action="store_true", help="Use a 2-layer MLP projector instead of linear.")
    parser.add_argument("--repa-input-size", type=int, default=224, help="Teacher input resolution (224 => 16x16 DINO patch grid).")
    parser.add_argument("--repa-r", type=int, default=1, help="Student block-call index r (1-based) to align at (fixed strategy).")
    parser.add_argument("--repa-r-strategy", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--repa-warmup-steps", type=int, default=0, help="Warmup steps for REPA λ.")
    parser.add_argument("--repa-stop-step", type=int, default=50_000, help="Step to stop applying REPA (early-stop). 0 => never stop.")
    parser.add_argument("--repa-decay-steps", type=int, default=0, help="Linear decay steps after stop-step to ramp λ to 0.")

    args = parser.parse_args()
    main(args)
