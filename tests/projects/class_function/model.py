from jaxtyping import Float
import torch
from torch import Tensor


class LinearModel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.zeros(out_features)

    def forward(self, x: Float[Tensor, "batch in"]) -> Float[Tensor, "batch out"]:
        """Forward pass."""
        out = x @ torch.transpose(self.weight, 0, 1) + self.bias
        return out
