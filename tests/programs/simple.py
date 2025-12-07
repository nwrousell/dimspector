from typing import Generic
import torch


class T(Generic(str)): ...


def broadcast(A: T["a b"], B: T["a 1"]):
    o = B.shape[0]
    t = B.shape[1]
    t = (t + 1) * 3
