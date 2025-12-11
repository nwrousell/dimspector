from typing import Generic
import torch


class T(Generic(str)): ...


def pool_project_wrong(x: T["batch seq d"], proj: T["seq out"]):
    pooled = torch.mean(x, dim=1)
    out = pooled @ proj
    return out
