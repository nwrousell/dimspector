from typing import Generic
import torch


class T(Generic(str)): ...


def empty():
    return


def single():
    1 + 2


def single_return():
    return 1 + 2


def params(a: T["a b"], b: T["b c"]):
    return a @ b


def if_else_join(a):
    if a:
        b = 1
    else:
        b = 2
    return b


def if_else_no_join(a):
    if a:
        return 1 + 2
    else:
        return 1 - 2


def method_vs_func(a):
    torch.ones(1, 2)
    a.reshape(-1)
