from jaxtyping import Float
import torch
from torch import Tensor


class AdaptiveLayer:
    def __init__(self, total_dim: int["total"]):
        self.total_dim = total_dim
        self.hidden_dim = total_dim // 2
        self.output_dim = total_dim - self.hidden_dim
        self.weight = torch.randn(self.hidden_dim, self.output_dim)

    def forward(self, x: Float[Tensor, "b total"]) -> Float[Tensor, "b output_dim"]:
        hidden = x[..., : self.hidden_dim]
        return hidden @ self.weight


def test_adaptive(x: Float[Tensor, "batch 256"]):
    base_dim = 128
    extra_dim = 64
    total = base_dim + extra_dim
    layer = AdaptiveLayer(total)
    y = layer.forward(x)
