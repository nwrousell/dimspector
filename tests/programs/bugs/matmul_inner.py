from jaxtyping import Float
import torch
from torch import Tensor


def mlp_wrong_hidden(x: Float[Tensor, "batch d"], w1: Float[Tensor, "d hidden"], w2: Float[Tensor, "other d"]):
    h = x @ w1
    out = h @ w2
    return out
