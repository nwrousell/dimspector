from jaxtyping import Float
import torch
from torch import Tensor


# VALID CASES - These should pass

def valid_int_param(k: int, x: Float[Tensor, "batch"]) -> Float[Tensor, "batch k"]:
    """k defined as int parameter (singleton)"""
    return x.repeat(k)


def valid_tensor_singletons(x: Float[Tensor, "d1 d2"]) -> Float[Tensor, "d1 d2"]:
    """d1 and d2 both appear as singletons in tensor parameter"""
    return x


def valid_multiple_params(
    batch: int, x: Float[Tensor, "batch d"], y: Float[Tensor, "batch d"]
) -> Float[Tensor, "batch d"]:
    """batch defined as int, d appears as singleton in both tensors"""
    return x + y


def valid_expression_with_singletons(
    k: int, x: Float[Tensor, "k"]
) -> Float[Tensor, "k"]:
    """k defined as int singleton, can be used in expressions"""
    return x


# INVALID CASES - These should fail with MissingSingletonDimVar error

def invalid_expression_only(x: Float[Tensor, "k*2"]) -> Float[Tensor, "k"]:
    """ERROR: k only appears in expression k*2, no singleton definition"""
    return x[:x.shape[0]//2]


def invalid_return_dimvar(k: int) -> Float[Tensor, "k m"]:
    """ERROR: m used in return but has no singleton definition"""
    return torch.randn(k, 5)


def invalid_param_expression(x: Float[Tensor, "batch k-1"]) -> Float[Tensor, "batch"]:
    """ERROR: k only in expression k-1, batch has no singleton"""
    return x


def invalid_compound_only(
    x: Float[Tensor, "d1*2"]
) -> Float[Tensor, "d1"]:
    """ERROR: d1 only appears in compound expression, no singleton"""
    return x[:x.shape[0]//2]


# CLASS CASES

class ValidModel(torch.nn.Module):
    def __init__(self, n: int, d: int):
        """Define n and d as singletons in __init__"""
        self.n = n
        self.d = d
        self.weight = torch.randn(d, n)

    def forward(self, x: Float[Tensor, "batch n"]) -> Float[Tensor, "batch d"]:
        """VALID: n and d defined in __init__ as int params, n appears as singleton"""
        return x @ torch.transpose(self.weight, 0, 1)

    def process(
        self, x: Float[Tensor, "batch n"], y: Float[Tensor, "batch d"]
    ) -> Float[Tensor, "batch n+d"]:
        """VALID: n and d from __init__, both appear as singletons in params"""
        return torch.cat([x, y], dim=1)


class InvalidModel(torch.nn.Module):
    def __init__(self, k: int):
        """Only k is defined in __init__"""
        self.k = k

    def forward(self, x: Float[Tensor, "batch"]) -> Float[Tensor, "batch m"]:
        """ERROR: m used in return but has no singleton definition"""
        return torch.randn(x.shape[0], 10)

    def process(self, x: Float[Tensor, "k*2"]) -> Float[Tensor, "p"]:
        """ERROR: p has no singleton definition anywhere"""
        return x[:5]


class AnotherInvalidModel(torch.nn.Module):
    def __init__(self, n: int):
        self.n = n

    def transform(self, x: Float[Tensor, "batch n+1"]) -> Float[Tensor, "batch q"]:
        """ERROR: q has no singleton definition"""
        return x[:, :7]
