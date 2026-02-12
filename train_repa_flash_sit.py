# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a research modification of the official DiT training script:
#   - Replaces diffusion scheduling/loss with SiT-style Transport (continuous-time interpolant).
#   - Adds optional REPA (REPresentation Alignment) regularization (from train_repa_flash.py)
#     using a frozen teacher encoder (DINOv2) and a small student projector + forward hooks.
#
# Intended to be used in the DiT/TRM codebase (latent-space ImageNet, SD VAE).

"""
A minimal training script for DiT using PyTorch DDP (SiT Transport + optional REPA).

Main loss:
  - transport.training_losses(model, x_lat, model_kwargs)["loss"]

Optional regularizer (REPA):
  - Align student token representations to frozen teacher patch tokens.

This script keeps the overall training skeleton identical to train_flash_sit.py and
mirrors the REPA mechanics from train_repa_flash.py.
"""

from __future__ import annotations

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

# -----------------------------------------------------------------------------
# Performance knobs (flash-friendly)
# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # Keep math SDP enabled as a fallback (REPA teacher can hit edge cases otherwise):
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# Model registry (flash variant):
from model_dittrm_xprev_sitvel import DiT_models  # type: ignore

# SiT transport:
from transport import create_transport  # type: ignore

# Frozen VAE:
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying nn.Module, unwrapping DDP and torch.compile wrappers."""
    if hasattr(m, "module"):
        m = m.module  # type: ignore[attr-defined]
    # torch.compile wraps modules in OptimizedModule with attribute _orig_mod
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod  # type: ignore[attr-defined]
    return m


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model towards the current model."""
    model = unwrap_model(model)
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Set requires_grad flag for all parameters in a model."""
    for p in model.parameters():
        p.requires_grad = flag


def cleanup() -> None:
    """End DDP training."""
    dist.destroy_process_group()


def create_logger(logging_dir: Optional[str]) -> logging.Logger:
    """Create a logger that writes to a log file and stdout."""
    if dist.get_rank() == 0:
        assert logging_dir is not None
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC,
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


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
            return base * (1.0 - float(step - stop) / float(decay))
        return 0.0
    return base


def preprocess_for_dino(x_img_m11: torch.Tensor, *, input_size: int) -> torch.Tensor:
    """
    x_img_m11: image tensor in [-1, 1], shape (B, 3, H, W)
    Returns normalized tensor for DINOv2 (ImageNet normalization).
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
    Returns patch tokens (B, N, D_t).

    If encoder_depth is provided and teacher exposes get_intermediate_layers(),
    interpret encoder_depth as 1-based layer index from start and take that layer.
    Otherwise, fall back to forward_features()["x_norm_patchtokens"].
    """
    if encoder_depth is not None and hasattr(teacher, "get_intermediate_layers"):
        idx0 = max(0, int(encoder_depth) - 1)
        try:
            if hasattr(teacher, "blocks"):
                if bool(getattr(teacher, "chunked_blocks", False)):
                    total_depth = len(teacher.blocks[-1])  # type: ignore[index]
                else:
                    total_depth = len(teacher.blocks)  # type: ignore[arg-type]
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
            return outs[0]
        except Exception:
            pass

    feats = teacher.forward_features(x_dino)
    return feats["x_norm_patchtokens"]


def _reshape_tokens_to_map(tokens: torch.Tensor) -> Optional[torch.Tensor]:
    """(B,N,D) -> (B,D,H,W) if N is a square; else None."""
    b, n, d = tokens.shape
    s = int(math.isqrt(n))
    if s * s != n:
        return None
    return tokens.transpose(1, 2).reshape(b, d, s, s)


def match_token_grid(teacher_tokens: torch.Tensor, student_tokens: torch.Tensor) -> torch.Tensor:
    """
    Ensure teacher token grid matches student token grid by interpolation if needed.
    Returns teacher_tokens_matched: (B, N_s, D_t).
    """
    if teacher_tokens.shape[1] == student_tokens.shape[1]:
        return teacher_tokens

    t_map = _reshape_tokens_to_map(teacher_tokens)
    s_map = _reshape_tokens_to_map(student_tokens)
    if t_map is None or s_map is None:
        t_global = teacher_tokens.mean(dim=1, keepdim=True)  # (B,1,D_t)
        return t_global.expand(-1, student_tokens.shape[1], -1).contiguous()

    _, _, hs, ws = s_map.shape
    t_map_rs = F.interpolate(t_map, size=(hs, ws), mode="bilinear", align_corners=False)
    t_tokens_rs = t_map_rs.flatten(2).transpose(1, 2).contiguous()
    return t_tokens_rs


def repa_cosine_loss(student_proj: torch.Tensor, teacher_tok: torch.Tensor) -> torch.Tensor:
    """Tokenwise cosine distance averaged over batch and tokens (computed in fp32)."""
    s = F.normalize(student_proj.float(), dim=-1)
    t = F.normalize(teacher_tok.float(), dim=-1)
    return (1.0 - (s * t).sum(dim=-1)).mean()


def attach_repa_projector(student: torch.nn.Module, teacher_dim: int, *, mlp: bool = False) -> None:
    """
    Attach a projector module to student as `student.repa_projector`, so it's saved in state_dict.
    """
    if not hasattr(student, "pos_embed"):
        raise AttributeError("Student model must have .pos_embed to infer hidden size for REPA projector.")
    d_student = int(student.pos_embed.shape[-1])  # type: ignore[attr-defined]

    if mlp:
        proj = torch.nn.Sequential(
            torch.nn.Linear(d_student, teacher_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(teacher_dim, teacher_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(teacher_dim, teacher_dim),
        )
    else:
        proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_student),
            torch.nn.Linear(d_student, teacher_dim),
        )
    student.repa_projector = proj  # type: ignore[attr-defined]


def setup_repa_hooks(student: torch.nn.Module) -> Tuple[Dict[str, object], list]:
    """
    Register forward hooks on blocks to capture the token tensor at block-call index r (1-based).
    Returns (state, handles).
    """
    state: Dict[str, object] = {"count": 0, "target": 1, "z": None}

    def _hook(_module, _inputs, output):
        state["count"] = int(state["count"]) + 1
        if int(state["count"]) == int(state["target"]):
            state["z"] = output
        return None

    handles = []
    if hasattr(student, "core_blocks"):
        blocks = list(student.core_blocks)  # type: ignore[attr-defined]
    elif hasattr(student, "blocks"):
        blocks = list(student.blocks)  # type: ignore[attr-defined]
    else:
        raise AttributeError("Student must have `.core_blocks` (TRM-DiT) or `.blocks` (vanilla DiT).")

    for blk in blocks:
        handles.append(blk.register_forward_hook(_hook))
    return state, handles


def _infer_trm_kwargs_from_ckpt_args(args: argparse.Namespace, ckpt_args: object) -> None:
    """
    Fill TRM architecture knobs from checkpoint args if user didn't pass them.
    (We use None defaults to detect "not specified".)
    """
    if ckpt_args is None:
        return
    for name in ["shared_depth", "trm_mode", "H_cycles", "L_cycles"]:
        if getattr(args, name) is None and hasattr(ckpt_args, name):
            setattr(args, name, getattr(ckpt_args, name))


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args: argparse.Namespace) -> None:
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # AMP config:
    amp_enabled = args.amp != "none"
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Model shape:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8."
    latent_size = args.image_size // 8

    # If resuming, load checkpoint early to infer TRM knobs:
    ckpt_obj = None
    ckpt_args_obj = None
    start_step = 0
    resumed = False
    resume_opt_state = None
    if args.ckpt is not None:
        ckpt_obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        if isinstance(ckpt_obj, dict):
            ckpt_args_obj = ckpt_obj.get("args", None)
        _infer_trm_kwargs_from_ckpt_args(args, ckpt_args_obj)

        # Recover global step:
        if isinstance(ckpt_obj, dict):
            start_step = ckpt_obj.get("train_steps", ckpt_obj.get("step", None))
        if start_step is None:
            base = os.path.basename(args.ckpt)
            stem = os.path.splitext(base)[0]
            digits = "".join([ch for ch in stem if ch.isdigit()])
            start_step = int(digits) if digits else 0
        resumed = True

    # Fill default TRM knobs if still None (training from scratch):
    if args.shared_depth is None:
        args.shared_depth = 1
    if args.trm_mode is None:
        args.trm_mode = "latent"
    if args.H_cycles is None:
        args.H_cycles = 3
    if args.L_cycles is None:
        args.L_cycles = 4

    # Create model (try TRM kwargs; fall back if ctor doesn't accept):
    model_ctor = DiT_models[args.model]
    ctor_kwargs = dict(
        input_size=latent_size,
        num_classes=args.num_classes,
        self_refine_feedback="xprev",
        prediction_type="velocity",
        feedback_dt=0.001,    
    )
    # Only include if available (won't pass None):
    if args.shared_depth is not None:
        ctor_kwargs["shared_depth"] = int(args.shared_depth)
    if args.trm_mode is not None:
        ctor_kwargs["trm_mode"] = str(args.trm_mode)
    if args.H_cycles is not None:
        ctor_kwargs["H_cycles"] = int(args.H_cycles)
    if args.L_cycles is not None:
        ctor_kwargs["L_cycles"] = int(args.L_cycles)

    # Training-time: do not early-stop
    ctor_kwargs["adaptive_halt"] = False
    # If your model supports this (as in train_repa_flash.py), keep it:
    ctor_kwargs["self_refine_feedback"] = "xprev"

    try:
        model_base = model_ctor(**ctor_kwargs)
    except TypeError:
        # Vanilla DiT fallback
        model_base = model_ctor(input_size=latent_size, num_classes=args.num_classes)

    # Optional REPA teacher + projector:
    teacher = None
    repa_state = None
    repa_hook_handles = None
    if args.use_repa:
        assert args.enc_type in _DINO_HUB_MAP, f"Unknown --enc-type {args.enc_type}."
        hub_name = _DINO_HUB_MAP[args.enc_type]
        logger.info(f"Loading REPA teacher via torch.hub: facebookresearch/dinov2::{hub_name}")
        teacher = torch.hub.load("facebookresearch/dinov2", hub_name)
        teacher.eval().to(device)
        requires_grad(teacher, False)
        if amp_dtype is not None:
            try:
                teacher.to(dtype=amp_dtype)
            except Exception:
                pass

        # Infer teacher dim on rank0 and broadcast:
        if rank == 0:
            dummy = torch.zeros(
                1, 3, args.repa_input_size, args.repa_input_size,
                device=device,
                dtype=(amp_dtype or torch.float32),
            )
            tok = dino_patch_tokens(teacher, dummy, encoder_depth=args.encoder_depth)
            teacher_dim = int(tok.shape[-1])
        else:
            teacher_dim = 0
        teacher_dim_t = torch.tensor([teacher_dim], device=device, dtype=torch.int64)
        dist.broadcast(teacher_dim_t, src=0)
        teacher_dim = int(teacher_dim_t.item())

        attach_repa_projector(model_base, teacher_dim, mlp=args.repa_proj_mlp)
        logger.info(
            f"Attached REPA projector: student_dim={int(model_base.pos_embed.shape[-1])} -> teacher_dim={teacher_dim}"
        )

    model_base = model_base.to(device)

    # EMA (uncompiled):
    ema = deepcopy(model_base).to(device)
    requires_grad(ema, False)

    # Optional torch.compile:
    if args.compile:
        try:
            model_base = torch.compile(model_base, mode=args.compile_mode)
        except TypeError:
            model_base = torch.compile(model_base)

    # Resume weights/opt:
    if resumed and isinstance(ckpt_obj, dict):
        # Load weights into underlying module (handles torch.compile wrappers)
        missing, unexpected = unwrap_model(model_base).load_state_dict(ckpt_obj["model"], strict=False)
        ema_missing, ema_unexpected = ema.load_state_dict(ckpt_obj["ema"], strict=False)
        resume_opt_state = ckpt_obj.get("opt", None)
        if rank == 0:
            logger.info(f"Resumed from {args.ckpt} at step={start_step}")
            if missing or unexpected:
                logger.info(f"[load_state_dict] model missing={len(missing)}, unexpected={len(unexpected)}")
            if ema_missing or ema_unexpected:
                logger.info(f"[load_state_dict] ema missing={len(ema_missing)}, unexpected={len(ema_unexpected)}")

        del ckpt_obj  # free CPU RAM

    # Wrap with DDP:
    model = DDP(
        model_base,
        device_ids=[device],
        find_unused_parameters=args.find_unused_parameters,
    )

    # REPA hooks on wrapped module:
    if args.use_repa:
        repa_state, repa_hook_handles = setup_repa_hooks(unwrap_model(model))
        logger.info("Registered REPA forward hooks on student blocks.")

    # Transport (SiT):
    transport = create_transport(
        path_type=args.path_type,
        prediction=args.prediction,
        loss_weight=args.loss_weight,
        train_eps=args.train_eps,
        sample_eps=args.sample_eps,
    )
    if rank == 0:
        logger.info(
            f"Transport: path_type={args.path_type}, prediction={args.prediction}, "
            f"loss_weight={args.loss_weight}, train_eps={transport.train_eps}, sample_eps={transport.sample_eps}"
        )

    # Frozen VAE:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    requires_grad(vae, False)
    if amp_dtype is not None:
        try:
            vae.to(dtype=amp_dtype)
        except Exception:
            pass

    logger.info(f"DiT Parameters: {sum(p.numel() for p in unwrap_model(model).parameters()):,}")

    # Optimizer:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    if resume_opt_state is not None:
        opt.load_state_dict(resume_opt_state)
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    data_sampler = DistributedSampler(
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
        sampler=data_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Initialize EMA from model weights only when starting fresh:
    if not resumed:
        update_ema(ema, model, decay=0.0)

    model.train()  # enable label dropout for CFG training
    ema.eval()
    if teacher is not None:
        teacher.eval()

    # Monitoring:
    train_steps = int(start_step)
    log_steps = 0
    running_total = 0.0
    running_main = 0.0
    running_repa = 0.0
    start_time = time()

    steps_per_epoch = len(loader)
    start_epoch = train_steps // steps_per_epoch
    skip_iters = train_steps % steps_per_epoch

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        data_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for it, (x, y) in enumerate(loader):
            if epoch == start_epoch and it < skip_iters:
                continue

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Keep copy for REPA teacher (still in [-1,1]):
            x_img = x

            # VAE encode to latents:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                    x_lat = vae.encode(x_img).latent_dist.sample().mul_(0.18215)

            model_kwargs = dict(y=y)

            # Reset REPA hook state for this forward:
            if args.use_repa and repa_state is not None:
                repa_state["count"] = 0
                repa_state["z"] = None

                if args.repa_r_strategy == "fixed":
                    target_r = int(args.repa_r)
                else:
                    student_mod = unwrap_model(model)
                    if hasattr(student_mod, "trm_mode") and str(getattr(student_mod, "trm_mode")) == "self_refine":
                        max_calls = int(getattr(student_mod, "H_cycles", 1)) * int(getattr(student_mod, "L_cycles", 1))
                    else:
                        max_calls = int(getattr(student_mod, "max_depth_steps", args.repa_r))
                    target_r = int(torch.randint(1, max_calls + 1, (1,), device=device).item())

                repa_state["target"] = target_r

            # Main Transport loss:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                loss_dict = transport.training_losses(model, x_lat, model_kwargs)
                loss_main = loss_dict["loss"].float().mean()

            loss = loss_main
            loss_repa = None
            lam = 0.0

            # Optional REPA regularizer:
            if args.use_repa and teacher is not None and repa_state is not None:
                lam = repa_lambda(
                    train_steps,
                    base=float(args.proj_coeff),
                    warmup=int(args.repa_warmup_steps),
                    stop=int(args.repa_stop_step),
                    decay=int(args.repa_decay_steps),
                )
                if lam > 0.0:
                    z = repa_state["z"]
                    if z is None:
                        raise RuntimeError(
                            "REPA hook did not capture student tokens. "
                            "Check that the model exposes core_blocks/blocks and repa_r is valid."
                        )

                    # Teacher tokens:
                    with torch.no_grad():
                        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                            x_dino = preprocess_for_dino(x_img, input_size=int(args.repa_input_size))
                            t_tok = dino_patch_tokens(teacher, x_dino, encoder_depth=args.encoder_depth)

                    # Student projection:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                        s_proj = unwrap_model(model).repa_projector(z)  # type: ignore[attr-defined]

                    # Match token grids if needed:
                    t_tok = match_token_grid(t_tok, s_proj)
                    if t_tok.shape[1] != s_proj.shape[1]:
                        t_tok = t_tok.mean(dim=1, keepdim=True).expand(-1, s_proj.shape[1], -1).contiguous()

                    loss_repa = repa_cosine_loss(s_proj, t_tok)
                    loss = loss + (lam * loss_repa)
                else:
                    # DDP stability: when λ=0, projector would be unused => touch it with 0-weight term.
                    if hasattr(unwrap_model(model), "repa_projector"):
                        proj = unwrap_model(model).repa_projector  # type: ignore[attr-defined]
                        dummy = torch.zeros((), device=device, dtype=loss_main.dtype)
                        for p in proj.parameters():
                            dummy = dummy + (p.sum() * 0.0)
                        loss = loss + dummy

            # Optim step:
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            update_ema(ema, model)

            # Logging:
            running_total += float(loss.item())
            running_main += float(loss_main.item())
            if loss_repa is not None:
                running_repa += float(loss_repa.item())
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / max(end_time - start_time, 1e-9)

                avg_total = torch.tensor(running_total / log_steps, device=device)
                avg_main = torch.tensor(running_main / log_steps, device=device)
                avg_repa = torch.tensor(running_repa / log_steps, device=device)

                dist.all_reduce(avg_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_main, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_repa, op=dist.ReduceOp.SUM)

                avg_total = avg_total.item() / dist.get_world_size()
                avg_main = avg_main.item() / dist.get_world_size()
                avg_repa = avg_repa.item() / dist.get_world_size()

                if args.use_repa:
                    logger.info(
                        f"(step={train_steps:07d}) "
                        f"Loss: {avg_total:.4f} (main={avg_main:.4f}, repa={avg_repa:.4f}, λ={lam:.3g}) "
                        f"Steps/Sec: {steps_per_sec:.2f}"
                    )
                else:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_total:.4f}, Steps/Sec: {steps_per_sec:.2f}")

                running_total = 0.0
                running_main = 0.0
                running_repa = 0.0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "train_steps": train_steps,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---------------- Standard DiT training args ----------------
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--ckpt", type=str, default=None, help="Resume training from a saved checkpoint (.pt).")

    # ---------------- Speed knobs ----------------
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="max-autotune")
    parser.add_argument("--find-unused-parameters", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20_000)

    # ---------------- TRM architecture knobs ----------------
    # Defaults are None so we can infer from checkpoint args if resuming.
    parser.add_argument("--shared-depth", dest="shared_depth", type=int, default=None)
    parser.add_argument("--trm-mode", dest="trm_mode", type=str, default=None, choices=["latent", "self_refine"])
    parser.add_argument("--H-cycles", dest="H_cycles", type=int, default=None)
    parser.add_argument("--L-cycles", dest="L_cycles", type=int, default=None)

    # ---------------- SiT transport / interpolant knobs ----------------
    parser.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    parser.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "noise", "score"])
    parser.add_argument("--loss-weight", type=str, default=None, choices=[None, "velocity", "likelihood"])
    parser.add_argument("--train-eps", type=float, default=None)
    parser.add_argument("--sample-eps", type=float, default=None)

    # ---------------- REPA knobs ----------------
    parser.add_argument("--use-repa", action="store_true", help="Enable REPA regularization.")
    parser.add_argument("--enc-type", type=str, default="dinov2-vit-b", choices=list(_DINO_HUB_MAP.keys()))
    parser.add_argument("--encoder-depth", type=int, default=8, help="Teacher layer index (1-based) if supported.")
    parser.add_argument("--proj-coeff", type=float, default=0.5, help="Base coefficient (λ) for REPA loss.")
    parser.add_argument("--repa-proj-mlp", action="store_true", help="Use a 2-layer MLP projector instead of linear.")
    parser.add_argument("--repa-input-size", type=int, default=224, help="Teacher input resolution.")
    parser.add_argument("--repa-r", type=int, default=1, help="Student block-call index r (1-based) to align at.")
    parser.add_argument("--repa-r-strategy", type=str, default="fixed", choices=["fixed", "random"])
    parser.add_argument("--repa-warmup-steps", type=int, default=0)
    parser.add_argument("--repa-stop-step", type=int, default=50_000, help="Stop applying REPA after this step. 0 => never.")
    parser.add_argument("--repa-decay-steps", type=int, default=0, help="Linear decay steps after stop-step to ramp λ->0.")

    main(parser.parse_args())
