from jaxtyping import Float
import torch
from torch import Tensor
from math_ops import broadcast_add, matmul, transpose, concat


def compute_matmul(
    x: Float[Tensor, "batch m k"], y: Float[Tensor, "batch k n"]
) -> Float[Tensor, "batch m n"]:
    """Compute matrix multiplication using matmul function."""
    return matmul(x, y)


def compute_transpose(x: Float[Tensor, "batch h w"]) -> Float[Tensor, "batch w h"]:
    """Compute transpose using transpose function."""
    return transpose(x)


def compute_broadcast(
    x: Float[Tensor, "batch dim"], y: Float[Tensor, "dim"]
) -> Float[Tensor, "batch dim"]:
    """Compute broadcast addition."""
    return broadcast_add(x, y)


def compute_concat(
    x: Float[Tensor, "batch dim1"], y: Float[Tensor, "batch dim2"]
) -> Float[Tensor, "batch dim1+dim2"]:
    """Concatenate tensors."""
    return concat(x, y)
