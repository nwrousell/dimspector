from typing import Generic
import torch


class T(Generic(str)): ...


def residual_dim_mismatch(x: T["batch d"], weight: T["d d_out"]):
    transformed = x @ weight
    out = x + transformed
    return out
