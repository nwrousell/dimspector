from typing import Generic
import torch


class T(Generic(str)): ...

def flatten(x: T["b h w"]):
    b = x.shape[0]
    x = torch.reshape(x, (b, -1))
    return x
