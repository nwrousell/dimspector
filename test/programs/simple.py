from typing import Generic
import torch


class T(Generic(str)): ...


def f(A: T["a b"]):
    B = A
    C = B + 2
    D = C @ B
    E = torch.ones(1, 2)
    E.reshape(-1)
    # if 1 < 2:
    #     B = A
    # else:
    #     C = B + 2
    # D = C @ B
