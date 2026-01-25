from jaxtyping import Float
import torch
from torch import Tensor


class PaddedLayer:
    def __init__(self, in_dim: int, padding: int):
        self.in_dim = in_dim
        self.padding = padding

        self.padded_dim = in_dim + 2 * padding
        self.weight = torch.randn(self.padded_dim, 64)

    def forward(self, x: Float[Tensor, "b in_dim"]) -> Float[Tensor, "b 64"]:
        padded = torch.nn.functional.pad(x, (self.padding, self.padding))
        return padded @ self.weight


def test_padded(x: Float[Tensor, "batch 128"]):
    layer = PaddedLayer(128, 4)
    y = layer.forward(x)
