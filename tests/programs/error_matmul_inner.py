from typing import Generic
import torch


class T(Generic(str)): ...


def mlp_wrong_hidden(x: T["batch d"], w1: T["d hidden"], w2: T["other d"]):
    """Error: w2's first dim should be 'hidden', not 'other'."""
    h = x @ w1
    out = h @ w2
    return out
