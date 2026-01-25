from jaxtyping import Float
import torch
from torch import Tensor

def concat(x: Float[Tensor, "m a"], y: Float[Tensor, "m b"], z: Float[Tensor, "m c"]):
    w = torch.concat((x, y, z), dim=1)
    return w
