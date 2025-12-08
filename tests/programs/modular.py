from typing import Generic
import torch


class T(Generic(str)): ...

def foo(x: T["a b"], y: T["b c"], z=True) -> T["a c"]:
    return x @ y

def bar(x: T["b b"]):
    z = foo(x, x)
    return z

# def bad_bingus(x: T["d e"], y: T["f g"]):
#     z = foo(x, y)
#     return z

def good_bingus(x: T["d e"], y: T["e g"]):
    z = foo(x, y)
    return z
