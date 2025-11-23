from typing import Generic
import torch


class T(Generic(str)): ...


def f(A: T["a b"]):
    B = A
    C = B + 2
    D = A @ B


# a = torch.zeros(5, 3)

# # binop
# # b = a - torch.ones(5, 3)

# # slicing/indexing
# b = a[0]

# # operations

# # reshaping
