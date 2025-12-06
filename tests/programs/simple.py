from typing import Generic
import torch


class T(Generic(str)): ...


def broadcast(A: T["a b"], B: T["a 1"]):
    C = A + B
