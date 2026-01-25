from jaxtyping import Float
import torch
from torch import Tensor


def pool_project_wrong(x: Float[Tensor, "batch seq d"], proj: Float[Tensor, "seq out"]):
    pooled = torch.mean(x, dim=1)
    out = pooled @ proj
    return out
