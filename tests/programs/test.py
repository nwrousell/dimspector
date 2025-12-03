from typing import Generic
import torch


class T(Generic(str)): ...


def f(a: T["a b"], b: T["a 1"]):
    c = a & b
    return c
