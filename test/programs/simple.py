from typing import Generic
import torch


class T(Generic(str)): ...


def f(A: T["a b"]):
    a = zeros(10, 20) if 2 else ones(20, 10)
    if 1 < 2:
        B = A
    else:
        C = B + 2
    D = C @ B


# a = torch.zeros(5, 3)

# # binop
# # b = a - torch.ones(5, 3)

# # slicing/indexing
# b = a[0]

# # operations

# # reshaping
