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


def simple_while(n):
    i = 0
    while i < n:
        i = i + 1
    return i


def simple_for(n):
    total = 0
    for i in range(n):
        total = total + i
    return total


def while_with_if(n):
    i = 0
    while i < n:
        if i % 2 == 0:
            i = i + 2
        else:
            i = i + 1
    return i


def nested_for(n, m):
    total = 0
    for i in range(n):
        for j in range(m):
            total = total + i + j
    return total


def nested_while(n, m):
    i = 0
    j = 0
    while i < n:
        while j < m:
            j = j + 1
        i = i + 1
        j = 0
    return i


def for_with_if(n):
    total = 0
    for i in range(n):
        if i > 5:
            total = total + i
        else:
            total = total + 1
    return total
