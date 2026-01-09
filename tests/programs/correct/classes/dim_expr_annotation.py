from jaxtyping import Float
import torch
from torch import Tensor


class ScaledLayer:
    def __init__(
        self,
        base_dim: int["base"],
        scale: int["scale"],
        scaled_dim: int["base * scale"],
    ):
        self.base_dim = base_dim
        self.scale = scale
        self.scaled_dim = scaled_dim
        self.weight = torch.randn(self.scaled_dim, 64)

    def forward(self, x: Float[Tensor, "b base"]) -> Float[Tensor, "b 64"]:
        scaled = x.repeat_interleave(self.scale, dim=-1)
        return scaled @ self.weight


def test_scaled(x: Float[Tensor, "batch 128"]):
    layer = ScaledLayer(128, 2, 256)
    y = layer.forward(x)
