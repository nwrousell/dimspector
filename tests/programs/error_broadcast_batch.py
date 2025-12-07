from typing import Generic
import torch


class T(Generic(str)): ...


def add_wrong_batch(x: T["batch seq d"], bias: T["other_batch d"]):
    """Error: bias has incompatible batch dim 'other_batch' instead of broadcastable shape."""
    out = x + bias
    return out
