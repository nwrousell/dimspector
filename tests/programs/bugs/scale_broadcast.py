from jaxtyping import Float
import torch
from torch import Tensor


def scale_wrong_channels(x: Float[Tensor, "batch channels h w"], scale: Float[Tensor, "other_channels"]):
    out = x * scale
    return out
