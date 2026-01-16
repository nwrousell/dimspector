from jaxtyping import Float
import torch
from torch import Tensor


class LinearModel:
    def __init__(self, in_features: int["in"], out_features: int["out"]):
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.zeros(out_features)

    def forward(self, x: Float[Tensor, "batch in"]) -> Float[Tensor, "batch out"]:
        """Forward pass."""
        return x @ torch.transpose(self.weight) + self.bias
