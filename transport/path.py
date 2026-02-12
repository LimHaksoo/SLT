import numpy as np
import torch as th

def expand_t_like_x(t: th.Tensor, x: th.Tensor) -> th.Tensor:
    """
    Reshape a 1D time tensor t: [B] to broadcast over x: [B, ...].
    """
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)

# -------------------- Coupling Plans / Paths --------------------

class ICPlan:
    """
    Linear Coupling Plan ("Linear" path in SiT codebases)

      x_t = alpha(t) * x1 + sigma(t) * x0
      alpha(t) = t
      sigma(t) = 1 - t

    where x0 ~ N(0, I) (noise), x1 is data.
    """
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        # coefficient on x1 and its derivative
        return t, 1.0

    def compute_sigma_t(self, t):
        # coefficient on x0 and its derivative
        return 1.0 - t, -1.0

    def compute_d_alpha_alpha_ratio_t(self, t):
        # d_alpha / alpha, used for drift computations
        return 1.0 / t

    def compute_drift(self, x, t):
        """
        Return (drift_mean, drift_var) in the *score* parametrization.

        This matches common SiT transport utilities.
        """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """
        Diffusion term g(t)^2 / 2 depending on form (used for SDE sampling).
        Not needed for training, but kept for completeness.
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1.0 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1.0) ** 2,
            "inccreasing-decreasing": norm * th.sin(np.pi * t) ** 2,
        }
        if form not in choices:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        return choices[form]

    # --- Conversion helpers between parameterizations ---

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x + 0.0  # alias
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_noise_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x + 0.0
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    # --- Path planning ---

    def compute_mu_t(self, t, x0, x1):
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class VPCPlan(ICPlan):
    """
    VP (variance preserving) path.

    This is a continuous-time VP-style interpolant used in SiT.
    """
    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: -0.25 * ((1.0 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1.0 - t) * self.sigma_min
        self.d_log_mean_coeff = lambda t: 0.5 * (1.0 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t):
        alpha_t = th.exp(self.log_mean_coeff(t))
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        p_sigma_t = 2.0 * self.log_mean_coeff(t)
        sigma_t = th.sqrt(1.0 - th.exp(p_sigma_t))
        d_sigma_t = th.exp(p_sigma_t) * (2.0 * self.d_log_mean_coeff(t)) / (-2.0 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1.0 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2.0


class GVPCPlan(ICPlan):
    """
    GVP path: alpha(t)=sin(pi t/2), sigma(t)=cos(pi t/2)
    """
    def __init__(self, sigma: float = 0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t):
        alpha_t = th.sin(t * np.pi / 2.0)
        d_alpha_t = (np.pi / 2.0) * th.cos(t * np.pi / 2.0)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = th.cos(t * np.pi / 2.0)
        d_sigma_t = -(np.pi / 2.0) * th.sin(t * np.pi / 2.0)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return np.pi / (2.0 * th.tan(t * np.pi / 2.0))
