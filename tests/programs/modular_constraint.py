from typing import Generic
import torch


class T(Generic(str)): ...

"""
in the following, should we either:
- have the user tell us b <=> c explicitly through the signature OR
- produce a constraint ourselves that b <=> c, which should be propagated
  to the caller and mapped back to their arguments' shapes
"""
def matmul(a: T["a b"], b: T["c d"]) -> T["a d"]:
    return a @ b

def f():
    x = torch.ones(( 5, 10)) # => Shape( 5, 10)
    y = torch.ones((15, 20)) # => Shape(20, 15)
    z = matmul(x, y)         # => deduced as bad because matmul expects b <=> c ==> 10 <=> 20 (?)
