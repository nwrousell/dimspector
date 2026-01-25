from jaxtyping import Float
import torch
from torch import Tensor


def add_wrong_batch(x: Float[Tensor, "batch seq d"], bias: Float[Tensor, "other_batch d"]):
    out = x + bias
    return out
