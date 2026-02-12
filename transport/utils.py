import torch as th

def mean_flat(x: th.Tensor) -> th.Tensor:
    """
    Mean over all non-batch dimensions.
    """
    return x.mean(dim=tuple(range(1, x.dim())))
