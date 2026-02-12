"""
Transport module (SiT-style) for training and sampling with interpolants.

Lightweight, self-contained port of the core pieces used for:
- scheduling (path planning)
- training losses (velocity/noise/score + optional reweighting)
- sampling (fixed-step ODE/SDE solvers)
"""
from __future__ import annotations

from .transport import Transport, ModelType, WeightType, PathType, Sampler


def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: str | None = None,
    train_eps: float | None = None,
    sample_eps: float | None = None,
) -> Transport:
    """
    Factory for Transport.

    Args:
      path_type: "Linear" | "GVP" | "VP"
      prediction: "velocity" | "noise" | "score"
      loss_weight: None | "velocity" | "likelihood"
      train_eps: epsilon away from endpoints used during training (to avoid singularities)
      sample_eps: epsilon away from endpoints used during sampling
    """
    # Prediction -> ModelType
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    # Loss reweighting -> WeightType
    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    # Path type string -> PathType
    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    if path_type not in path_choice:
        raise ValueError(f"Unknown path_type={path_type} (choose from {list(path_choice.keys())})")
    path_enum = path_choice[path_type]

    # Default eps handling (mirrors common SiT implementations)
    if path_enum in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_enum in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0.0 if train_eps is None else train_eps
        sample_eps = 0.0 if sample_eps is None else sample_eps

    return Transport(
        model_type=model_type,
        path_type=path_enum,
        loss_type=loss_type,
        train_eps=float(train_eps),
        sample_eps=float(sample_eps),
    )
