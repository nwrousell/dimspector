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

"""
in the following, in a world w/ constraints, if the user tells us b has shape [b, c], we may under the hood
still give a and b distinct dimvars [a, b], [c, d], but just immediately introduce the constraint
b <=> c b/c the user typed b as [b, c]
"""
def matmul_explicit(a: T["a b"], b: T["b c"]) -> T["a c"]:
    return a @ b

def f():
    x = torch.ones(( 5, 10)) # => Shape( 5, 10)
    y = torch.ones((15, 20)) # => Shape(15, 20)
    z = matmul(x, y)         # => deduced as bad because matmul expects b <=> c ==> 10 <=> 15 (?)
