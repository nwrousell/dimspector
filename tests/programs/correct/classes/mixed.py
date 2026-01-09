from jaxtyping import Float
import torch
from torch import Tensor


class ComplexLayer:
    def __init__(
        self,
        input_tensor: Float[Tensor, "c h w"],
        scale: int["scale"],
        bias: int["bias"],
    ):
        self.input_shape = input_tensor
        self.scale = scale
        self.bias = bias
        # Extract dims from tensor
        c, h, w = input_tensor.shape
        # Use dimvar expressions
        self.scaled_h = h * scale
        self.scaled_w = w * scale
        self.weight = torch.randn(c, self.scaled_h, self.scaled_w)

    def forward(
        self, x: Float[Tensor, "b c h w"]
    ) -> Float[Tensor, "b c h*scale w*scale"]:
        # Upsample using scale
        upsampled = torch.nn.functional.interpolate(
            x, size=(self.scaled_h, self.scaled_w)
        )
        return upsampled @ self.weight.unsqueeze(0)


def test_complex(x: Float[Tensor, "batch 3 16 16"]):
    first = x[0]
    layer = ComplexLayer(first, 2, 1)
    y = layer.forward(x)
