from jaxtyping import Float
import torch
from torch import Tensor


class PaddedConvLayer:
    def __init__(
        self,
        in_dim: int,
        padding: int,
        padded_dim: int,
        kernel_size: int,
        out_dim: int,
    ):
        self.in_dim = in_dim
        self.padding = padding
        self.padded_dim = padded_dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.weight = torch.randn(self.kernel_size, self.out_dim)

    def forward(self, x: Float[Tensor, "b in_dim"]) -> Float[Tensor, "b out_dim"]:
        padded = torch.nn.functional.pad(x, (self.padding, self.padding))
        return padded[:, : self.out_dim] @ self.weight


def test_padded_conv(x: Float[Tensor, "batch 128"]):
    layer = PaddedConvLayer(128, 4, 136, 3, 134)
    y = layer.forward(x)
