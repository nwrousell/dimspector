from typing import Generic
import torch


class T(Generic(str)): ...


def pool_project_wrong(x: T["batch seq d"], proj: T["seq out"]):
    """Error: after mean over dim=1, shape is [batch, d], but proj expects [seq, out]."""
    pooled = torch.mean(x, dim=1)
    out = pooled @ proj
    return out
