from typing import Generic
import torch


class T(Generic(str)): ...

def concat(x: T["m a"], y: T["m b"], z: T["m c"]):
    w = torch.concat((x, y, z), dim=1)
    return w
