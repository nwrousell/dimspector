from jaxtyping import Float
import torch
from torch import Tensor

class LinearModel:
    def __init__(self, in_features: int["in"], out_features: int["out"]):
        self.weight = torch.randn(out_features, in_features)
        self.bias = torch.zeros(out_features)

    def forward(self, x: Float[Tensor, "b in"]) -> Float[Tensor, "b out"]:
        """Forward pass."""
        return x @ torch.transpose(self.weight) + self.bias

class MultiLayerModel:
    def __init__(self, in_features: int["in"], hidden: int["hidden"], out_features: int["out"]):
        self.layers = [
            LinearModel(in_features, hidden),
            LinearModel(hidden, hidden),
            LinearModel(hidden, hidden),
            LinearModel(hidden, out_features),
        ]

    def forward(self, x: Float[Tensor, "batch in"]) -> Float[Tensor, "batch out"]:
        x = self.layers[0].forward(x)
        x = self.layers[1].forward(x)
        x = self.layers[2].forward(x)
        x = self.layers[3].forward(x)
        return x
