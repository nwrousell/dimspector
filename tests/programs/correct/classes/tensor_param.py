from jaxtyping import Float
import torch
from torch import Tensor


class ConvLayer:
    def __init__(self, input_shape: Float[Tensor, "c h w"], out_channels: int["out_c"]):
        self.input_shape = input_shape
        self.out_channels = out_channels
        c, h, w = input_shape.shape
        self.weight = torch.randn(out_channels, c, 3, 3)

    def forward(self, x: Float[Tensor, "b c h w"]) -> Float[Tensor, "b out_c h w"]:
        return torch.conv2d(x, self.weight)


def test_conv(x: Float[Tensor, "batch 3 32 32"]):
    conv = ConvLayer(x[0], 64)
    y = conv.forward(x)
