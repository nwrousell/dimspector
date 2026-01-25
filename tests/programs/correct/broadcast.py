from jaxtyping import Float
import torch
from torch import Tensor


def broadcast(A: Float[Tensor, "a b"], B: Float[Tensor, "a 1"]):
    o = B.shape[0]
    t = B.shape[1]
    t = (t + 1) * 3
