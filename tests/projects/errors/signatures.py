from jaxtyping import Float
import torch
from torch import Tensor


def require_same_k(
    x: Float[Tensor, "batch k"], y: Float[Tensor, "batch k"]
) -> Float[Tensor, "batch k"]:
    return x + y


def require_3d(
    x: Float[Tensor, "batch height width"],
) -> Float[Tensor, "batch height width"]:
    return x


def require_2d(x: Float[Tensor, "m n"]) -> Float[Tensor, "m n"]:
    return x


def require_same_dim(
    a: Float[Tensor, "n"], b: Float[Tensor, "n"]
) -> Float[Tensor, "n"]:
    return a + b


def with_expression(
    x: Float[Tensor, "batch k"], y: Float[Tensor, "batch k-1"]
) -> Float[Tensor, "batch k"]:
    return x


class ShapeChecker:
    def check_dims(
        self, x: Float[Tensor, "batch d"], y: Float[Tensor, "batch d"]
    ) -> Float[Tensor, "batch d"]:
        return x + y

    def matmul_wrapper(
        self, x: Float[Tensor, "batch m k"], y: Float[Tensor, "batch k n"]
    ) -> Float[Tensor, "batch m n"]:
        return x @ y
