# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import os
os.environ["TORCH_ALLOW_CUBLAS_LT"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
try:
    torch.backends.cuda.preferred_blas_library("cublas")
except Exception:
    pass
# # the first flag below was False when we tested this script but True makes A100 training a lot faster:
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# Optional: prefer FlashAttention / mem-efficient SDPA when available (PyTorch 2.x).
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    # Disable math SDP to encourage fused kernels (will fall back automatically if unsupported):
    torch.backends.cuda.enable_math_sdp(False)
except Exception:
    pass

# # Optional: new matmul precision API (keeps backward compatibility).
# try:
#     torch.set_float32_matmul_precision("high")
# except Exception:
#     pass

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from model_dittrm_flash import DiT_models
# from models import DiT_models

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

import torch.nn as nn
import torch.nn.functional as F


class ConvProjAttnProcessor2_0:
    """
    cuBLAS/cuBLASLt가 불안정한 환경(SM75 등)에서 VAE attention의 Q/K/V/to_out 선형프로젝션이
    cublasLtMatmul/cublas*Batched GEMM 경로로 들어가며 크래시가 나는 경우가 있습니다.

    이 processor는 Q/K/V/to_out[0] 선형프로젝션을 (B, S, C) -> (B, C, S)로 전치한 뒤
    1x1 Conv1d(F.conv1d)로 계산해 (대부분 cuDNN conv 경로)로 우회합니다.
    """
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        # (B, C, H, W) -> (B, S, C)
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2).contiguous()
        else:
            b, s, c = hidden_states.shape
            h = w = None
            hidden_states = hidden_states.contiguous()

        # Optional norms (match diffusers processors)
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2).contiguous()

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif getattr(attn, "norm_cross", False):
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # helper: linear projection via 1x1 conv1d (avoid GEMM)
        def proj_conv1d(x_bsc, linear):
            # x_bsc: (B, S, C) float32
            x = x_bsc.transpose(1, 2).contiguous()               # (B, C, S)
            w = linear.weight.unsqueeze(-1).contiguous()         # (O, C, 1)
            b = linear.bias.contiguous() if linear.bias is not None else None
            y = F.conv1d(x, w, b)                                # (B, O, S)
            return y.transpose(1, 2).contiguous()                # (B, S, O)

        # Q/K/V
        q = proj_conv1d(hidden_states, attn.to_q)
        k = proj_conv1d(encoder_hidden_states, attn.to_k)
        v = proj_conv1d(encoder_hidden_states, attn.to_v)

        inner_dim = q.shape[-1]
        head_dim = inner_dim // attn.heads

        # (B, S, O) -> (B, H, S, D)
        q = q.view(b, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        k = k.view(b, -1, attn.heads, head_dim).transpose(1, 2).contiguous()
        v = v.view(b, -1, attn.heads, head_dim).transpose(1, 2).contiguous()

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, k.shape[-2], b)
            attention_mask = attention_mask.view(b, attn.heads, -1, attention_mask.shape[-1])

        hidden_states = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(b, -1, attn.heads * head_dim).contiguous()

        # out proj via conv1d too
        hidden_states = proj_conv1d(hidden_states, attn.to_out[0])
        hidden_states = attn.to_out[1](hidden_states)

        # back to (B, C, H, W) if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(b, c, h, w).contiguous()

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states


class FlatProjAttnProcessor2_0:
    """
    Attention processor for Diffusers Attention blocks that forces 2D GEMMs for
    Q/K/V and output projections by flattening (B, S, C) -> (B*S, C).
    This avoids cuBLAS strided-batched GEMM paths that can be unstable on some
    older GPUs / CUDA builds during VAE encoding.
    """
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kwargs):
        residual = hidden_states
        input_ndim = hidden_states.ndim

        # (B, C, H, W) -> (B, S, C)
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2).contiguous()
        else:
            b, s, c = hidden_states.shape
            h = w = None
            hidden_states = hidden_states.contiguous()

        # optional norms (as in diffusers processors)
        if getattr(attn, "spatial_norm", None) is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        if getattr(attn, "group_norm", None) is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2).contiguous()

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            encoder_hidden_states = encoder_hidden_states.contiguous()
            if getattr(attn, "norm_cross", False):
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # ---- Q/K/V projections: force 2D GEMM ----
        bq, sq, cq = hidden_states.shape
        hs2 = hidden_states.reshape(bq * sq, cq).contiguous()
        q = attn.to_q(hs2).view(bq, sq, -1)

        bk, sk, ck = encoder_hidden_states.shape
        ehs2 = encoder_hidden_states.reshape(bk * sk, ck).contiguous()
        k = attn.to_k(ehs2).view(bk, sk, -1)
        v = attn.to_v(ehs2).view(bk, sk, -1)

        inner_dim = k.shape[-1]
        head_dim = inner_dim // attn.heads

        q = q.view(bq, sq, attn.heads, head_dim).transpose(1, 2)  # (B, H, S, D)
        k = k.view(bk, sk, attn.heads, head_dim).transpose(1, 2)
        v = v.view(bk, sk, attn.heads, head_dim).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sk, bq)
            attention_mask = attention_mask.view(bq, attn.heads, -1, attention_mask.shape[-1])

        hidden_states = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(bq, sq, attn.heads * head_dim).contiguous()

        # ---- output projection: force 2D GEMM ----
        out2 = hidden_states.reshape(bq * sq, -1).contiguous()
        hidden_states = attn.to_out[0](out2).view(bq, sq, -1)
        hidden_states = attn.to_out[1](hidden_states)

        # (B, S, C) -> (B, C, H, W)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(1, 2).reshape(b, c, h, w).contiguous()

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / getattr(attn, "rescale_output_factor", 1.0)
        return hidden_states
def unwrap_model(m):
    """Return the underlying nn.Module, unwrapping DDP and torch.compile wrappers."""
    if hasattr(m, "module"):
        m = m.module
    # torch.compile wraps modules in OptimizedModule with attribute _orig_mod
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    model = unwrap_model(model)
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if name in ema_params:
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
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
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

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor

    # Create an EMA of the (uncompiled) model for evaluation/checkpointing:
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # AMP config (FlashAttention kernels typically require fp16/bf16 to trigger):
    amp_enabled = args.amp != "none"
    if args.amp == "fp16":
        amp_dtype = torch.float16
    elif args.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    # Optional torch.compile (compile the training model, not EMA):
    if args.compile:
        try:
            model = torch.compile(model, mode=args.compile_mode)
        except TypeError:
            # Older torch.compile signatures may not accept mode=...
            model = torch.compile(model)

    # start_step = 0
    # if args.ckpt is not None:
    #     # 1) CPU로 로드해서 OOM 방지 (PR에서도 이 부분이 문제라고 언급됨)
    #     ckpt = torch.load(args.ckpt, map_location="cpu")  # :contentReference[oaicite:4]{index=4}

    #     # 2) state_dict 복구 (키 이름은 ckpt.keys()로 확인해서 맞추세요)
    #     model.load_state_dict(ckpt["model"])
    #     ema.load_state_dict(ckpt["ema"])
    #     optimizer.load_state_dict(ckpt["opt"])

    #     # 3) optimizer state 텐서를 GPU로 옮기기 (이거 안 하면 느리거나 에러/메모리 이슈)
    #     for state in optimizer.state.values():
    #         for k, v in state.items():
    #             if torch.is_tensor(v):
    #                 state[k] = v.cuda()

    #     # 4) step 복구 (checkpoint에 저장된 키 이름에 맞게)
    #     start_step = ckpt.get("train_steps", ckpt.get("step", 0))
    #     print(f"Resumed from {args.ckpt} at step={start_step}")
    #     del ckpt
    
    start_step = 0
    resumed = False
    resume_opt_state = None
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

        # Load weights into the underlying module (handles torch.compile wrappers).
        unwrap_model(model).load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        resume_opt_state = ckpt.get("opt", None)

        # Recover global step (older checkpoints may not store it; fall back to filename).
        start_step = ckpt.get("train_steps", ckpt.get("step", None))
        if start_step is None:
            base = os.path.basename(args.ckpt)
            stem = os.path.splitext(base)[0]
            digits = "".join([ch for ch in stem if ch.isdigit()])
            start_step = int(digits) if digits else 0

        resumed = True
        if rank == 0:
            logger.info(f"Resumed from {args.ckpt} at step={start_step}")
        del ckpt

    # Wrap with DDP:
    model = DDP(
        model,
        device_ids=[device],
        find_unused_parameters=args.find_unused_parameters,
    )

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    # VAE is frozen:
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval().requires_grad_(False)
    vae.to(device, dtype=torch.float32)
    requires_grad(vae, False)
    vae.enable_slicing()
    vae.set_attn_processor(ConvProjAttnProcessor2_0())
    # if amp_dtype is not None:
    #     try:
    #         vae.to(dtype=amp_dtype)
    #     except Exception:
    #         pass

    logger.info(f"DiT Parameters: {sum(p.numel() for p in unwrap_model(model).parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if resume_opt_state is not None:
        opt.load_state_dict(resume_opt_state)
        # Move optimizer state tensors to the correct device.
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    if not resumed:
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    # Prepare models for training:
    # update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = start_step
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    steps_per_epoch = len(loader)
    start_epoch = start_step // steps_per_epoch
    skip_iters = start_step % steps_per_epoch

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for it, (x, y) in enumerate(loader):
            # if epoch == start_epoch and it < skip_iters:
            #     continue
            # Move batch to GPU:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Map input images to latent space + normalize latents (VAE is frozen):
            with torch.no_grad():
                x_fp32 = x.float()
                with torch.autocast(device_type="cuda", enabled=False):
                    x_latent = vae.encode(x_fp32).latent_dist.sample()
                    x_latent = x_latent.mul_(0.18215)

            # DiT는 AMP dtype으로 계속 (fp16 권장)
            x = x_latent.to(dtype=amp_dtype) if amp_dtype is not None else x_latent

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            # DiT forward + diffusion loss under autocast:
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].float().mean()

            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
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
    # Speed knobs:
    parser.add_argument("--amp", type=str, choices=["none", "fp16", "bf16"], default="bf16",
                        help="Automatic mixed precision dtype. bf16 is recommended on Ampere+ GPUs.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile for the training model (first iteration will be slower).")
    parser.add_argument("--compile-mode", type=str, default="max-autotune",
                        help="torch.compile mode (e.g., default, reduce-overhead, max-autotune).")
    parser.add_argument("--find-unused-parameters", action="store_true",
                        help="DDP find_unused_parameters=True (useful for debugging).")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=20_000)
    parser.add_argument("--ckpt", type=str, default=None, help="resume training from a saved checkpoint")
    args = parser.parse_args()
    main(args)
