
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file is a **research modification** of the official DiT `models.py`:
#   "Scalable Diffusion Models with Transformers" (DiT).
#
# Goal: Provide a TRM-style (Tiny Recursive Model) *recurrent / weight-tied* DiT variant
# that reuses 1 (or a small number) of Transformer blocks for multiple "depth steps".
#
# Drop-in usage:
#   - Place this file as `models.py` in the DiT repo (or import it similarly).
#   - It exposes the same `DiT_*` constructors and `DiT_models` dict.
#
# Notes:
#   - We keep DiT's public API and training/sampling expectations intact:
#       forward(x, t, y) -> model_out with shape [B, out_channels, H, W]
#     where out_channels = in_channels (epsilon) or 2*in_channels (learn_sigma).
#   - We add recurrence controls via kwargs (see DiTTRM.__init__ docstring).
#
# References:
#   - DiT: https://github.com/facebookresearch/DiT
#   - TRM (Tiny Recursive Models): https://github.com/SamsungSAILMontreal/TinyRecursiveModels

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Optional dependency: timm. DiT originally uses timm's PatchEmbed/Attention/Mlp.
# For portability (and for environments where timm isn't installed), we provide
# minimal fallbacks that match the interfaces DiT expects.
# ---------------------------------------------------------------------

try:
    from timm.models.vision_transformer import PatchEmbed, Attention, Mlp  # type: ignore
except Exception:  # pragma: no cover
    class PatchEmbed(nn.Module):
        """Minimal ViT PatchEmbed fallback.

        Input:  (B, C, H, W)
        Output: (B, N, D) where N = (H/ps)*(W/ps)
        """
        def __init__(self, img_size=32, patch_size=2, in_chans=4, embed_dim=768, bias=True):
            super().__init__()
            if isinstance(img_size, tuple):
                img_h, img_w = img_size
            else:
                img_h = img_w = int(img_size)
            if isinstance(patch_size, tuple):
                ph, pw = patch_size
            else:
                ph = pw = int(patch_size)

            assert img_h % ph == 0 and img_w % pw == 0, "img_size must be divisible by patch_size"
            self.img_size = (img_h, img_w)
            self.patch_size = (ph, pw)
            self.grid_size = (img_h // ph, img_w // pw)
            self.num_patches = self.grid_size[0] * self.grid_size[1]

            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(ph, pw), stride=(ph, pw), bias=bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, H, W)
            x = self.proj(x)  # (B, D, H/ps, W/ps)
            x = x.flatten(2).transpose(1, 2)  # (B, N, D)
            return x

    class Attention(nn.Module):
        """Minimal multi-head self-attention fallback compatible with timm.Attention."""
        def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
            super().__init__()
            assert dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = self.head_dim ** -0.5

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = float(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, N, C = x.shape
            qkv = self.qkv(x)  # (B, N, 3C)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

            # PyTorch 2.x fast path:
            if hasattr(F, "scaled_dot_product_attention"):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_drop if self.training else 0.0,
                    is_causal=False
                )  # (B, heads, N, head_dim)
            else:  # pragma: no cover
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                if self.attn_drop and self.training:
                    attn = F.dropout(attn, p=self.attn_drop)
                out = attn @ v

            out = out.transpose(1, 2).reshape(B, N, C)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

    class Mlp(nn.Module):
        """Minimal MLP fallback compatible with timm.Mlp."""
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop = nn.Dropout(drop)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x


# ---------------------------------------------------------------------
# Core DiT building blocks (same as official, with minor refactors)
# ---------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation primitive used by DiT."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param t: (B,) 1-D Tensor of timesteps (may be fractional).
        :param dim: embedding dimension.
        :param max_period: controls min frequency.
        :return: (B, dim) embedding tensor.
        """
        # Adapted from GLIDE.
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations (supports CFG via label dropout)."""
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + int(use_cfg_embedding), hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Drops labels to enable classifier-free guidance."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, torch.tensor(self.num_classes, device=labels.device), labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class DiTBlock(nn.Module):
    """
    A DiT block with adaLN-Zero conditioning.
    Same math as official DiT.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        # Produce shift/scale/gates for MSA and MLP.
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """The final DiT layer: adaLN + linear projection back to patch pixels (epsilon/sigma)."""
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------
# TRM-style DiT (weight-tied / recurrent) — drop-in replacement
# ---------------------------------------------------------------------

class DiTTRM(nn.Module):
    """
    TRM-style DiT: reuse a small set of transformer blocks repeatedly.

    Parameters match official DiT, plus recurrence knobs:

    Recurrence knobs (all optional via kwargs):
      - shared_depth (int): number of *unique* blocks to instantiate (default=1).
                            The network cycles through these blocks while iterating.
      - depth (int): maximum number of recurrent applications (R_max). We preserve
                     the original `depth` argument name for compatibility.
      - trm_mode (str): "latent" or "self_refine".
            * "latent": (A) repeat blocks on internal tokens; predict epsilon once at end.
            * "self_refine": (B) TRM-like outer loop that updates an epsilon estimate `y`
                             and feeds it back.
      - H_cycles (int): number of outer improvement steps (K) for self_refine (default=3).
      - L_cycles (int): number of inner latent updates per improvement step (n) for self_refine (default=4).
                        Total compute ~ H_cycles * L_cycles.
      - adaptive_halt (bool): enable early stopping at inference-time (default=True).
      - halt_eps (float): halting threshold on mean absolute delta of epsilon between steps.
                          (default=1e-3)
      - min_steps (int): minimum number of steps before halting can trigger (default=1).
      - inject_x (bool): inject x_t tokens into the latent update every step (default=True).
      - inject_y (bool): inject current epsilon tokens y into latent update (default=True for self_refine).
      - inject_scale_init (Tuple[float,float]): initial (x_scale, y_scale) for learnable injection scalars.

    Output:
      - Same as DiT: (B, out_channels, H, W)
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        # --- TRM / recurrence extras ---
        shared_depth: int = 1,
        trm_mode: str = "latent",
        H_cycles: int = 3,
        L_cycles: int = 4,
        adaptive_halt: bool = True,
        halt_eps: float = 1e-3,
        min_steps: int = 1,
        inject_x: bool = True,
        inject_y: Optional[bool] = None,
        inject_scale_init: Tuple[float, float] = (1.0, 0.0),
        self_refine_feedback: str = "xprev",
        # --- prediction semantics / feedback update (for SiT / flow-matching style training) ---
        # "eps" (default): diffusion epsilon prediction (DiT default)
        # "velocity": continuous-time velocity prediction (e.g., SiT / flow matching)
        prediction_type: str = "eps",
        # Optional dt used when prediction_type=="velocity" and self_refine_feedback uses xprev/x0.
        # If None, we fall back to 1/(T-1) where T=diffusion schedule length (default 1000).
        feedback_dt: Optional[float] = None,
        # Direction for xprev update when using velocity: "decrease" means x_{t-dt} = x_t - dt*v
        feedback_direction: str = "decrease",
        **block_kwargs,
    ):
        super().__init__()

        assert trm_mode in ["latent", "self_refine"], f"Unknown trm_mode={trm_mode}"

        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma

        # Base embeddings (same as DiT):
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob) if num_classes > 0 else None

        # Fixed sinusoidal positional embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)

        # Weight-tied / recurrent core blocks:
        assert shared_depth >= 1
        self.shared_depth = shared_depth
        self.max_depth_steps = int(depth)  # keep name consistent with official
        self.core_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs)
            for _ in range(shared_depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # TRM-loop settings:
        self.trm_mode = trm_mode
        self.H_cycles = int(H_cycles)
        self.L_cycles = int(L_cycles)
        self.adaptive_halt = bool(adaptive_halt)
        self.halt_eps = float(halt_eps)
        self.min_steps = int(min_steps)

        self.inject_x = bool(inject_x)
        if inject_y is None:
            inject_y = (trm_mode == "self_refine")
        self.inject_y = bool(inject_y)

        # Learnable scalar gates to control how much x/y are injected into the evolving latent.
        x_scale0, y_scale0 = inject_scale_init
        self.inject_scale_x = nn.Parameter(torch.tensor(float(x_scale0)))
        self.inject_scale_y = nn.Parameter(torch.tensor(float(y_scale0)))

        # Self-refinement feedback type:
        #  - "eps": feed back epsilon (baseline behavior)
        #  - "xprev": convert eps -> x_{t-1} (DDIM-style) and feed back x_{t-1} tokens
        #  - "x0": convert eps -> x0 and feed back x0 tokens
        self.self_refine_feedback = str(self_refine_feedback)
        if self.self_refine_feedback not in ["eps", "xprev", "x0"]:
            raise ValueError(
                f"Unknown self_refine_feedback={self.self_refine_feedback}. Use one of: eps, xprev, x0"
            )

        # Prediction semantics:
        #   - "eps": DiT / diffusion default (model predicts epsilon)
        #   - "velocity": SiT / flow-matching style (model predicts dx/dt a.k.a. velocity)
        pred_type = str(prediction_type).lower()
        if pred_type in ["eps", "epsilon"]:
            self.prediction_type = "eps"
        elif pred_type in ["v", "vel", "velocity"]:
            self.prediction_type = "velocity"
        else:
            raise ValueError(f"Unknown prediction_type={prediction_type}. Use 'eps' or 'velocity'.")

        # Feedback dt (only used for velocity -> xprev/x0 conversions). If None, inferred from schedule length.
        self.feedback_dt = float(feedback_dt) if feedback_dt is not None else None

        # Feedback direction (for velocity updates):
        dir_ = str(feedback_direction).lower()
        if dir_ not in ["decrease", "increase"]:
            raise ValueError(f"Unknown feedback_direction={feedback_direction}. Use 'decrease' or 'increase'.")
        self.feedback_direction = dir_

        # Default diffusion schedule buffers (DiT default: 1000-step linear betas).
        # Used only when self_refine_feedback != "eps" (eps <-> x0/xprev conversions).
        self._init_default_diffusion_schedule(num_timesteps=1000)

        # Diagnostics (populated during forward):
        self.last_num_steps: Optional[int] = None

        self.initialize_weights()

    # -----------------------------
    # Initialization (same spirit as official DiT)
    # -----------------------------
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Positional embedding: fixed sin-cos.
        gs = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], gs)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # PatchEmbed conv init (match MAE/DiT style):
        if hasattr(self.x_embedder, "proj") and isinstance(self.x_embedder.proj, nn.Conv2d):
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if self.x_embedder.proj.bias is not None:
                nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Label embedding init:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Timestep embedding init:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation and final layer to start near-identity / zero output:
        for blk in self.core_blocks:
            nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # -----------------------------
    # Token <-> image utilities
    # -----------------------------
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, p*p*C)
        return: (B, C, H, W)
        """
        p = self.patch_size
        b, t, d = x.shape
        c = self.out_channels
        h = w = int(t ** 0.5)
        assert h * w == t, "Number of patches must be a square."
        x = x.reshape(b, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(b, c, h * p, w * p)
        return imgs

    def _embed_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed an image/latent tensor to tokens and add pos-embed.
        """
        x_tokens = self.x_embedder(x)  # (B, T, D)
        return x_tokens + self.pos_embed

    def _make_condition(self, t: torch.Tensor, y: Optional[torch.Tensor]) -> torch.Tensor:
        t_emb = self.t_embedder(t)
        if self.y_embedder is not None and y is not None:
            y_emb = self.y_embedder(y, self.training)
            return t_emb + y_emb
        return t_emb


    # -----------------------------
    # Diffusion schedule helpers (for eps <-> x0/xprev conversions in self-refine)
    # -----------------------------
    def _init_default_diffusion_schedule(
        self,
        *,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        """Initialize a default linear beta schedule (DiT / guided-diffusion default).

        We store sqrt(ᾱ_t) and sqrt(1-ᾱ_t) as buffers for fast extraction.
        If you use a different diffusion schedule, call `set_diffusion_schedule(betas=...)` once
        after creating your diffusion object.
        """
        betas = torch.linspace(float(beta_start), float(beta_end), int(num_timesteps), dtype=torch.float64)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_ab = torch.sqrt(alphas_cumprod).float()
        sqrt_omab = torch.sqrt(1.0 - alphas_cumprod).float()

        # Register buffers only once; afterwards, copy_ to keep state_dict stable.
        if not hasattr(self, "sqrt_alphas_cumprod"):
            self.register_buffer("sqrt_alphas_cumprod", sqrt_ab, persistent=False)
            self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_omab, persistent=False)
        else:
            if self.sqrt_alphas_cumprod.numel() != sqrt_ab.numel():
                raise ValueError(
                    "Diffusion schedule length mismatch; recreate the model for a different num_timesteps."
                )
            self.sqrt_alphas_cumprod.data.copy_(sqrt_ab.to(self.sqrt_alphas_cumprod.dtype))
            self.sqrt_one_minus_alphas_cumprod.data.copy_(sqrt_omab.to(self.sqrt_one_minus_alphas_cumprod.dtype))

    @torch.no_grad()
    def set_diffusion_schedule(self, *, betas: torch.Tensor) -> None:
        """Override the diffusion schedule buffers with an explicit beta tensor.

        Args:
          betas: shape (T,), dtype float32/float64, on CPU or GPU.
        """
        betas = betas.detach().float().cpu()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_ab = torch.sqrt(alphas_cumprod)
        sqrt_omab = torch.sqrt(1.0 - alphas_cumprod)

        if not hasattr(self, "sqrt_alphas_cumprod"):
            # If schedule buffers were not initialized yet, init with correct length.
            self.register_buffer("sqrt_alphas_cumprod", sqrt_ab, persistent=False)
            self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_omab, persistent=False)
            return

        if self.sqrt_alphas_cumprod.numel() != sqrt_ab.numel():
            raise ValueError(
                f"Schedule length mismatch: model has T={self.sqrt_alphas_cumprod.numel()}, provided T={sqrt_ab.numel()}. "
                "Recreate the model if you changed num_timesteps."
            )
        self.sqrt_alphas_cumprod.data.copy_(sqrt_ab.to(self.sqrt_alphas_cumprod.dtype))
        self.sqrt_one_minus_alphas_cumprod.data.copy_(sqrt_omab.to(self.sqrt_one_minus_alphas_cumprod.dtype))

    def _extract(self, arr_1d: torch.Tensor, t: torch.Tensor, x_like: torch.Tensor) -> torch.Tensor:
        """Extract values from a 1-D schedule tensor at timesteps t and reshape for broadcasting."""
        if t.dtype != torch.long:
            t = t.long()
        t = t.clamp(0, arr_1d.shape[0] - 1)
        out = arr_1d.to(device=x_like.device, dtype=x_like.dtype).gather(0, t)
        while out.ndim < x_like.ndim:
            out = out.view(-1, *([1] * (x_like.ndim - 1)))
        return out

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        a_t = self._extract(self.sqrt_alphas_cumprod, t, x_t)
        b_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return (x_t - b_t * eps) / (a_t + 1e-8)

    def _predict_xprev_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM-style one-step-less-noisy latent using the *same* eps."""
        t_prev = (t.long() - 1).clamp(min=0)
        a_prev = self._extract(self.sqrt_alphas_cumprod, t_prev, x_t)
        b_prev = self._extract(self.sqrt_one_minus_alphas_cumprod, t_prev, x_t)
        x0_hat = self._predict_x0_from_eps(x_t, t, eps)
        return a_prev * x0_hat + b_prev * eps


    # -----------------------------
    # Continuous-time helpers (for SiT / flow-matching style training)
    # -----------------------------
    def _t_to_unit_interval(self, t: torch.Tensor) -> torch.Tensor:
        """Convert `t` to a float tensor roughly in [0, 1].

        Heuristics:
          - If `t` is integer timesteps (0..T-1), map to [0,1] via t/(T-1).
          - If `t` is float but max(t)>1.5, assume it's still in 0..T-1 scale and map to [0,1].
          - Otherwise assume it's already in [0,1].

        This is only used for *feedback* conversions when `prediction_type == "velocity"`.
        """
        T = int(self.sqrt_alphas_cumprod.numel()) if hasattr(self, "sqrt_alphas_cumprod") else 1000
        if t.dtype in (torch.int32, torch.int64, torch.long):
            return t.float() / float(max(1, T - 1))
        t_f = t.float()
        if t_f.detach().max() > 1.5:
            return t_f / float(max(1, T - 1))
        return t_f

    def _infer_feedback_dt(self, t: torch.Tensor, dt: Optional[float] = None) -> float:
        """Pick a dt for velocity-based xprev updates.

        Priority:
          1) `dt` passed to forward(...)
          2) `self.feedback_dt` from constructor
          3) default: 1/(T-1) where T is schedule length (default 1000 -> ~0.001)
        """
        if dt is not None:
            return float(dt)
        if getattr(self, "feedback_dt", None) is not None:
            return float(self.feedback_dt)
        T = int(self.sqrt_alphas_cumprod.numel()) if hasattr(self, "sqrt_alphas_cumprod") else 1000
        return 1.0 / float(max(1, T - 1))

    def _predict_xprev_from_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor, *, dt: Optional[float] = None
    ) -> torch.Tensor:
        """First-order (Euler) estimate of x_{t-dt} from velocity v = dx/dt.

        If `feedback_direction == "decrease"` (default):
            x_prev = x_t - dt * v

        If `feedback_direction == "increase"`:
            x_prev = x_t + dt * v
        """
        dt_val = self._infer_feedback_dt(t, dt)
        sign = -1.0 if self.feedback_direction == "decrease" else 1.0
        return x_t + (sign * dt_val) * v

    def _predict_x0_from_velocity(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Estimate x0 assuming a *linear* interpolant in unit time: x(t) = x0 + t*v.

        This is exact for straight-line paths used in many flow-matching / interpolant setups.
        If your transport path is different, you may want to override this with your path-specific inversion.
        """
        t_u = self._t_to_unit_interval(t)
        while t_u.ndim < x_t.ndim:
            t_u = t_u.view(-1, *([1] * (x_t.ndim - 1)))
        return x_t - t_u * v

    # -----------------------------
    # Recurrent core
    # -----------------------------
    def _apply_recurrent_blocks(self, z: torch.Tensor, c: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Apply recurrent transformer blocks for `steps` iterations.
        """
        for i in range(steps):
            blk = self.core_blocks[i % self.shared_depth]
            z = blk(z, c)
        return z

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        feedback_dt: Optional[float] = None,
        return_all_eps: bool = False,
    ):
        """
        Forward pass.

        Args:
          x: (B, C, H, W) latent input x_t
          t: (B,) timesteps
          y: (B,) class labels (optional)
          return_all_eps: if True, returns (final_out, eps_list) for debugging.

        Returns:
          model_out: (B, out_channels, H, W)
        """
        # Embed fixed input x_t:
        x_tokens = self._embed_input(x)
        c = self._make_condition(t, y)

        eps_list: List[torch.Tensor] = []

        if self.trm_mode == "latent":
            # ---------------------------------------------------------
            # (A) Latent-only recurrence:
            #    z0 = x_tokens, then z_{k+1} = Block(z_k, c)
            #    output = Head(z_R, c)
            # Optionally halt early based on epsilon deltas (inference).
            # ---------------------------------------------------------
            z = x_tokens

            prev_eps: Optional[torch.Tensor] = None
            steps_taken = 0

            for step in range(self.max_depth_steps):
                # One recurrent step:
                blk = self.core_blocks[step % self.shared_depth]
                z_in = z
                if self.inject_x:
                    z_in = z_in + self.inject_scale_x * x_tokens
                # no y injection in latent mode by default
                z = blk(z_in, c)
                steps_taken += 1

                # Optional halting based on predicted epsilon changes:
                if self.adaptive_halt and (not self.training):
                    # compute epsilon at this step cheaply (head is linear)
                    out_tokens = self.final_layer(z, c)
                    out_img = self.unpatchify(out_tokens)
                    eps = out_img[:, : self.in_channels] if self.learn_sigma else out_img
                    eps_list.append(eps)

                    if prev_eps is not None and steps_taken >= self.min_steps:
                        delta = (pred - prev_eps).abs().mean().item()
                        if delta < self.halt_eps:
                            break
                    prev_eps = pred

            # Final prediction:
            out_tokens = self.final_layer(z, c)
            model_out = self.unpatchify(out_tokens)

            self.last_num_steps = steps_taken
            if return_all_eps:
                return model_out, eps_list
            return model_out

        # ---------------------------------------------------------
        # (B) Self-refinement TRM-style:
        #    Keep x_tokens fixed.
        #    Maintain y_tokens (epsilon estimate) and latent z.
        #    Outer loop: H_cycles "improvement steps".
        #      - Inner loop: L_cycles updates to z (reasoning).
        #      - Update y (epsilon) from z via head; feed back next outer step.
        #    Halt early if epsilon changes are small (inference).
        # ---------------------------------------------------------
        z = x_tokens  # latent state
        # Initialize feedback image for y_tokens.
        #   - eps: start from 0 (baseline TRM-style)
        #   - xprev/x0: start from x_t (in-domain, avoids eps-domain mismatch)
        if self.self_refine_feedback == "eps":
            y_img = torch.zeros(x.shape[0], self.in_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        else:
            y_img = x
        y_tokens = self._embed_input(y_img)

        prev_eps: Optional[torch.Tensor] = None
        steps_taken = 0

        for h in range(self.H_cycles):
            # Thinking: update z for L_cycles.
            for l in range(self.L_cycles):
                blk = self.core_blocks[(steps_taken) % self.shared_depth]
                z_in = z
                if self.inject_x:
                    z_in = z_in + self.inject_scale_x * x_tokens
                if self.inject_y:
                    z_in = z_in + self.inject_scale_y * y_tokens
                z = blk(z_in, c)
                steps_taken += 1

            # Answer update: produce epsilon (and maybe sigma) from z.
            out_tokens = self.final_layer(z, c)
            out_img = self.unpatchify(out_tokens)

            pred = out_img[:, : self.in_channels] if self.learn_sigma else out_img
            eps_list.append(pred)

            # Update feedback tokens for the next outer cycle.
            # The model's head predicts either:
            #   - epsilon (diffusion)  if prediction_type == "eps"
            #   - velocity (SiT/flow)  if prediction_type == "velocity"
            # but we can choose an *in-domain* feedback variable to reduce distribution shift.
            if self.self_refine_feedback == "eps":
                # Feed back the raw prediction (eps or velocity).
                y_img = pred
            elif self.self_refine_feedback == "x0":
                # Feed back a denoised / clean-domain estimate.
                if self.prediction_type == "velocity":
                    y_img = self._predict_x0_from_velocity(x, t, pred)
                else:
                    y_img = self._predict_x0_from_eps(x, t, pred)
            else:  # "xprev"
                # Feed back an "x_prev" estimate in the same domain as x_t.
                if self.prediction_type == "velocity":
                    y_img = self._predict_xprev_from_velocity(x, t, pred, dt=feedback_dt)
                else:
                    y_img = self._predict_xprev_from_eps(x, t, pred)
            y_tokens = self._embed_input(y_img)

            # Inference-time halting:
            if self.adaptive_halt and (not self.training):
                if prev_eps is not None and (h + 1) >= self.min_steps:
                    delta = (pred - prev_eps).abs().mean().item()
                    if delta < self.halt_eps:
                        break
                prev_eps = pred

        self.last_num_steps = steps_taken
        # Return the final out_img which includes sigma if learn_sigma:
        if return_all_eps:
            return out_img, eps_list
        return out_img

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        """
        Forward pass of DiT with classifier-free guidance.
        Matches the official DiT implementation style.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        # Apply CFG to first 3 channels by default (official DiT behavior).
        # For alternative behavior, modify this split.
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ---------------------------------------------------------------------
# Positional embedding helpers (from MAE)
# ---------------------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0) -> np.ndarray:
    """
    grid_size: int of the grid height and width
    return:
      pos_embed: (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim) if cls_token
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w, h
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000 ** omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# ---------------------------------------------------------------------
# Factory functions (same names/keys as official DiT)
# Here, they create DiTTRM by default (drop-in replacement).
# You can switch back to the original DiT by editing these to return the baseline.
# ---------------------------------------------------------------------

def DiT_XL_2(**kwargs): return DiTTRM(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
def DiT_XL_4(**kwargs): return DiTTRM(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
def DiT_XL_8(**kwargs): return DiTTRM(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):  return DiTTRM(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)
def DiT_L_4(**kwargs):  return DiTTRM(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)
def DiT_L_8(**kwargs):  return DiTTRM(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):  return DiTTRM(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)
def DiT_B_4(**kwargs):  return DiTTRM(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)
def DiT_B_8(**kwargs):  return DiTTRM(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):  return DiTTRM(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)
def DiT_S_4(**kwargs):  return DiTTRM(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)
def DiT_S_8(**kwargs):  return DiTTRM(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

# Official-style model registry:
DiT_models: Dict[str, callable] = {
    "DiT-XL/2": DiT_XL_2, "DiT-XL/4": DiT_XL_4, "DiT-XL/8": DiT_XL_8,
    "DiT-L/2":  DiT_L_2,  "DiT-L/4":  DiT_L_4,  "DiT-L/8":  DiT_L_8,
    "DiT-B/2":  DiT_B_2,  "DiT-B/4":  DiT_B_4,  "DiT-B/8":  DiT_B_8,
    "DiT-S/2":  DiT_S_2,  "DiT-S/4":  DiT_S_4,  "DiT-S/8":  DiT_S_8,
}
