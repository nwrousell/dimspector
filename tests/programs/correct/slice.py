from jaxtyping import Float
import torch
from torch import Tensor

def token_targets(tokens: Float[Tensor, "b t"]):
    inps = tokens[:1, :-1]
    targets = tokens[:,1:]
    z = inps + targets
    return inps, targets
