from jaxtyping import Float
import torch
from torch import Tensor


def residual_dim_mismatch(x: Float[Tensor, "batch d"], weight: Float[Tensor, "d d_out"]):
    transformed = x @ weight
    out = x + transformed
    return out
