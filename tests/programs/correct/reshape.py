from jaxtyping import Float
import torch
from torch import Tensor

def flatten(x: Float[Tensor, "b h w"]):
    b = x.shape[0]
    x = torch.reshape(x, (b, -1))
    return x
