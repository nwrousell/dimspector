from jaxtyping import Float
import torch
from torch import Tensor


def add(
    x: Float[Tensor, "batch dim"], y: Float[Tensor, "batch dim"]
) -> Float[Tensor, "batch dim"]:
    """Add two tensors."""
    return x + y


def broadcast_add(
    x: Float[Tensor, "batch dim"], y: Float[Tensor, "dim"]
) -> Float[Tensor, "batch dim"]:
    """Add with broadcasting."""
    return x + y


def matmul(
    x: Float[Tensor, "batch m k"], y: Float[Tensor, "batch k n"]
) -> Float[Tensor, "batch m n"]:
    """Matrix multiplication."""
    return x @ y


def transpose(x: Float[Tensor, "batch h w"]) -> Float[Tensor, "batch w h"]:
    """Transpose last two dimensions."""
    return torch.transpose(x, -1, -2)


def concat(
    x: Float[Tensor, "batch dim1"], y: Float[Tensor, "batch dim2"]
) -> Float[Tensor, "batch dim1+dim2"]:
    """Concatenate along last dimension."""
    return torch.cat([x, y], dim=-1)
