from jaxtyping import Float
import torch
from torch import Tensor


def init_weight_matrix(
    in_features: int, out_features: int
) -> Float[Tensor, "out in"]:
    """Initialize a weight matrix with Kaiming initialization."""
    weight = torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
    weight.requires_grad_(True)
    return weight


def init_bias_vector(out_features: int) -> Float[Tensor, "out"]:
    """Initialize a bias vector with zeros."""
    bias = torch.zeros(out_features)
    bias.requires_grad_(True)
    return bias
