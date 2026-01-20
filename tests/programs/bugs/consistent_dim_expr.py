from jaxtyping import Float
import torch
from torch import Tensor


def process_with_reduced_dim(
    x: Float[Tensor, "batch d_model-1"],
    weights: Float[Tensor, "d_model-1 out"],
    bias: Float[Tensor, "d_model-1"],
):
    result = x @ weights + bias
    return result
