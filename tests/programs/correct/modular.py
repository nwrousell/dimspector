from jaxtyping import Float
import torch
from torch import Tensor

def foo(x: Float[Tensor, "a b"], y: Float[Tensor, "b c"], z=True) -> Float[Tensor, "a c"]:
    return x @ y

def car(x: Float[Tensor, "b d"]) -> Float[Tensor, "b d-1"]:
    x = x[:,:-1]
    return x

def tar(x: Float[Tensor, "b d"], y: Float[Tensor, "b-1 d"]) -> Float[Tensor, "b d"]:
    return x

def bar(x: Float[Tensor, "b b"]):
    z = foo(x, x)
    return z

def baz(x: Float[Tensor, "h w"]):
    x = car(x)
    return x

def quz(x: Float[Tensor, "b d"]):
    y = x[:-1,:]
    z = tar(x, y)
    return z

# def bad_bingus(x: Float[Tensor, "d e"], y: Float[Tensor, "f g"]):
#     z = foo(x, y)
#     return z

def good_bingus(x: Float[Tensor, "d e"], y: Float[Tensor, "e g"]):
    z = foo(x, y)
    return z
