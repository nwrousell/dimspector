from typing import Generic
import torch


class T(Generic(str)): ...

def foo(x: T["a b"], y: T["b c"], z=True) -> T["a c"]:
    return x @ y

def car(x: T["b d"]) -> T["b d-1"]:
    x = x[:,:-1]
    return x

def tar(x: T["b d"], y: T["b-1 d"]) -> T["b d"]:
    return x

def bar(x: T["b b"]):
    z = foo(x, x)
    return z

def baz(x: T["h w"]):
    x = car(x)
    return x

def quz(x: T["b d"]):
    y = x[:-1,:]
    z = tar(x, y)
    return z

# def bad_bingus(x: T["d e"], y: T["f g"]):
#     z = foo(x, y)
#     return z

def good_bingus(x: T["d e"], y: T["e g"]):
    z = foo(x, y)
    return z
