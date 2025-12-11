from typing import Generic
import torch


class T(Generic(str)): ...


def add_wrong_batch(x: T["batch seq d"], bias: T["other_batch d"]):
    out = x + bias
    return out
