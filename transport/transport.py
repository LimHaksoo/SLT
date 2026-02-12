import enum
import math
from typing import Any, Dict, Optional

import torch as th

from . import path
from .utils import mean_flat


class ModelType(enum.Enum):
    """What the model predicts."""
    VELOCITY = enum.auto()
    NOISE = enum.auto()
    SCORE = enum.auto()


class PathType(enum.Enum):
    """Which interpolant path / coupling plan."""
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """Loss re-weighting type."""
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    """
    Core SiT-style "transport" object:
      - chooses a path sampler (Linear / GVP / VP)
      - defines training losses for velocity/noise/score prediction
      - provides drift() / score() helpers for ODE/SDE sampling
    """

    def __init__(
        self,
        *,
        model_type: ModelType = ModelType.VELOCITY,
        path_type: PathType = PathType.LINEAR,
        loss_type: WeightType = WeightType.NONE,
        train_eps: float = 0.0,
        sample_eps: float = 0.0,
    ):
        self.model_type = model_type
        self.path_type = path_type
        self.loss_type = loss_type
        self.train_eps = float(train_eps)
        self.sample_eps = float(sample_eps)

        # Choose path sampler implementation.
        if path_type == PathType.LINEAR:
            self.path_sampler = path.ICPlan()
        elif path_type == PathType.GVP:
            self.path_sampler = path.GVPCPlan()
        elif path_type == PathType.VP:
            self.path_sampler = path.VPCPlan()
        else:
            raise ValueError(f"Unknown path_type={path_type}")

    # ------------------------------------------------------------------
    # Time interval helpers
    # ------------------------------------------------------------------

    def check_interval(
        self,
        train_eps: float,
        sample_eps: float,
        *,
        diffusion_form: str = "SBDM",
        sde: bool = False,
        reverse: bool = False,
        eval: bool = False,
        last_step_size: float = 0.0,
    ):
        """
        Match common SiT implementations: avoid endpoints for certain paths / model types.

        Returns (t0, t1) within [0,1].
        """
        t0 = 0.0
        t1 = 1.0
        eps = train_eps if not eval else sample_eps

        # VP path becomes ill-conditioned near t=1.
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1.0 - eps if (not sde or last_step_size == 0) else 1.0 - last_step_size

        # Linear/GVP: if using noise/score (or SDE), avoid t=0 and t=1 by eps.
        elif (
            type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]
            and (self.model_type != ModelType.VELOCITY or sde)
        ):
            # For SBDM SDE, or non-velocity models, take t0=eps.
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0.0
            t1 = 1.0 - eps if (not sde or last_step_size == 0) else 1.0 - last_step_size

        if reverse:
            t0, t1 = 1.0 - t0, 1.0 - t1

        return float(t0), float(t1)

    # ------------------------------------------------------------------
    # Training-time sampling and loss
    # ------------------------------------------------------------------

    def sample(self, x1: th.Tensor):
        """
        Draw x0 ~ N(0,I) and t ~ Uniform([t0,t1]) based on x1 shape.
        """
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],), device=x1.device, dtype=x1.dtype) * (t1 - t0) + t0
        return t, x0, x1

    def training_losses(self, model, x1: th.Tensor, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Compute SiT-style training loss terms.

        IMPORTANT: This expects the model to output either:
          - C channels (same as x1), OR
          - 2C channels (learn_sigma), in which case we slice to first C.
        """
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        model_output = model(xt, t, **model_kwargs)
        # Handle learn_sigma (DiT-style): model_output may be [B,2C,H,W]
        if model_output.shape[1] == 2 * xt.shape[1]:
            model_output = model_output[:, : xt.shape[1]]

        assert model_output.shape == xt.shape, (
            f"Transport expects model output shape == xt shape. "
            f"Got model_output={tuple(model_output.shape)} vs xt={tuple(xt.shape)}"
        )

        if self.model_type == ModelType.VELOCITY:
            loss = mean_flat((model_output - ut) ** 2)
            return {'loss': loss, 'pred': model_output}

        # Score/noise training uses drift_var / sigma_t weighting
        _, drift_var = self.path_sampler.compute_drift(xt, t)
        sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))

        if self.loss_type == WeightType.VELOCITY:
            weight = (drift_var / sigma_t) ** 2
        elif self.loss_type == WeightType.LIKELIHOOD:
            weight = drift_var / (sigma_t ** 2)
        elif self.loss_type == WeightType.NONE:
            weight = 1.0
        else:
            raise NotImplementedError(f"Unknown loss_type={self.loss_type}")

        if self.model_type == ModelType.NOISE:
            loss = mean_flat(weight * ((model_output - x0) ** 2))
        elif self.model_type == ModelType.SCORE:
            # Score target is -x0/sigma_t -> equivalently enforce model_output*sigma_t + x0 = 0
            loss = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
        else:
            raise NotImplementedError(f"Unknown model_type={self.model_type}")

        return {'loss': loss, 'pred': model_output}

    # ------------------------------------------------------------------
    # Sampling-time helpers
    # ------------------------------------------------------------------

    def prior_logp(self, z: th.Tensor) -> th.Tensor:
        """
        Log-density under standard normal prior: z ~ N(0,I), per batch element.
        Useful for likelihood evaluation (optional).
        """
        B = z.shape[0]
        dim = int(th.prod(th.tensor(z.shape[1:], device=z.device)))
        # -0.5 * (dim*log(2pi) + ||z||^2)
        return -0.5 * (dim * math.log(2.0 * math.pi) + (z.reshape(B, -1) ** 2).sum(dim=1))

    def get_drift(self):
        """
        Return drift function for the probability flow ODE.

        Signature: drift(x, t, model, **model_kwargs) -> Tensor same shape as x.
        """
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output)

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        elif self.model_type == ModelType.VELOCITY:
            drift_fn = velocity_ode
        else:
            raise NotImplementedError(f"Unknown model_type={self.model_type}")

        def body_fn(x, t, model, **model_kwargs):
            out = drift_fn(x, t, model, **model_kwargs)
            assert out.shape == x.shape, "Output shape from ODE solver must match input shape"
            return out

        return body_fn

    def get_score(self):
        """
        Return score function for x_t = alpha_t * x1 + sigma_t * x0.

        Signature: score(x, t, model, **model_kwargs) -> Tensor same shape as x.
        """
        if self.model_type == ModelType.NOISE:
            def score_fn(x, t, model, **kwargs):
                sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
                return model(x, t, **kwargs) / -sigma_t
        elif self.model_type == ModelType.SCORE:
            def score_fn(x, t, model, **kwargs):
                return model(x, t, **kwargs)
        elif self.model_type == ModelType.VELOCITY:
            def score_fn(x, t, model, **kwargs):
                v = model(x, t, **kwargs)
                return self.path_sampler.get_score_from_velocity(v, x, t)
        else:
            raise NotImplementedError(f"Unknown model_type={self.model_type}")

        return score_fn


class ode:
    """
    Lightweight fixed-step ODE integrator.

    Supports:
      - Euler
      - Heun (explicit trapezoid / improved Euler)
    """

    def __init__(
        self,
        *,
        drift,
        t0: float,
        t1: float,
        sampler_type: str = "Euler",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        self.drift = drift
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.sampler_type = sampler_type
        self.num_steps = int(num_steps)
        self.atol = float(atol)
        self.rtol = float(rtol)

        st = sampler_type.lower()
        if st not in ["euler", "heun"]:
            raise NotImplementedError(
                f"Only fixed-step ODE solvers are implemented (Euler/Heun). Got sampler_type={sampler_type}."
            )

    @th.no_grad()
    def sample(self, init: th.Tensor, model, **model_kwargs) -> th.Tensor:
        x = init
        dt = (self.t1 - self.t0) / float(self.num_steps)
        t = self.t0

        for _ in range(self.num_steps):
            t_batch = th.full((x.shape[0],), t, device=x.device, dtype=x.dtype)

            if self.sampler_type.lower() == "euler":
                dx = self.drift(x, t_batch, model, **model_kwargs)
                x = x + dx * dt

            elif self.sampler_type.lower() == "heun":
                k1 = self.drift(x, t_batch, model, **model_kwargs)
                x_pred = x + k1 * dt
                t2 = t + dt
                t2_batch = th.full((x.shape[0],), t2, device=x.device, dtype=x.dtype)
                k2 = self.drift(x_pred, t2_batch, model, **model_kwargs)
                x = x + (k1 + k2) * (0.5 * dt)

            t += dt

        return x


class sde:
    """
    Lightweight Euler-Maruyama SDE integrator.

    We treat `diffusion_fn(x,t)` as the diffusion coefficient g(t) in:
        dX = f(X,t) dt + g(X,t) dW
    """

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0: float,
        t1: float,
        num_steps: int = 250,
        sampler_type: str = "Euler",
    ):
        self.drift = drift
        self.diffusion = diffusion
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.num_steps = int(num_steps)
        self.sampler_type = sampler_type

        st = sampler_type.lower()
        if st not in ["euler"]:
            raise NotImplementedError(f"Only Euler-Maruyama is implemented. Got sampler_type={sampler_type}.")

    @th.no_grad()
    def sample(self, init: th.Tensor, model, **model_kwargs) -> th.Tensor:
        x = init
        dt = (self.t1 - self.t0) / float(self.num_steps)
        t = self.t0

        sqrt_dt = math.sqrt(dt) if dt > 0 else 0.0

        for _ in range(self.num_steps):
            t_batch = th.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            drift = self.drift(x, t_batch, model, **model_kwargs)

            diff = self.diffusion(x, t_batch)  # should broadcast over x
            # Ensure diffusion broadcasts to x shape:
            if diff.dim() == 1:
                diff = path.expand_t_like_x(diff, x)
            elif diff.shape != x.shape:
                # Try standard broadcasting:
                diff = diff

            noise = th.randn_like(x)
            x = x + drift * dt + diff * sqrt_dt * noise

            t += dt

        return x


class Sampler:
    """Sampler class for the transport model (ODE / SDE)."""

    def __init__(self, transport: Transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(self, *, diffusion_form: str = "SBDM", diffusion_norm: float = 1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        sde_drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(
            x, t, model, **kwargs
        )
        sde_diffusion = diffusion_fn
        return sde_drift, sde_diffusion

    def __get_last_step(self, sde_drift, *, last_step: Optional[str], last_step_size: float):
        if last_step is None:
            return lambda x, t, model, **kwargs: x
        if last_step == "Mean":
            return lambda x, t, model, **kwargs: x + sde_drift(x, t, model, **kwargs) * last_step_size
        if last_step == "Euler":
            return lambda x, t, model, **kwargs: x + self.drift(x, t, model, **kwargs) * last_step_size
        if last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **kwargs: x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(
                x, t, model, **kwargs
            )
        raise NotImplementedError(f"Unknown last_step={last_step}")

    def sample_sde(
        self,
        *,
        sampling_method: str = "Euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: Optional[str] = None,
        last_step_size: float = 0.0,
        num_steps: int = 250,
    ):
        """
        Return a function that samples using an SDE solver.

        If last_step is not None, we integrate up to t1=1-last_step_size and then apply last_step to reach t=1.
        """
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(sde_drift, sde_diffusion, t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method)
        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            x = _sde.sample(init, model, **model_kwargs)
            if last_step_size > 0.0:
                ts = th.ones(init.size(0), device=init.device, dtype=init.dtype) * t1
                x = last_step_fn(x, ts, model, **model_kwargs)
            return x

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method: str = "Euler",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
    ):
        """
        Return a function that samples using a fixed-step ODE solver (Euler or Heun).
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1.0 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        _ode = ode(drift=drift, t0=t0, t1=t1, sampler_type=sampling_method, num_steps=num_steps, atol=atol, rtol=rtol)

        return _ode.sample
