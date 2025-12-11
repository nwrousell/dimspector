from typing import Generic
import torch


class T(Generic(str)): ...

def token_targets(tokens: T["b t"]):
    inps = tokens[:1, :-1]
    targets = tokens[:,1:]
    z = inps + targets
    return inps, targets
