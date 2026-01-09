from jaxtyping import Float
import torch
from torch import Tensor


class MyDense:
    def __init__(self, in_dim: int["in_dim"], out_dim: int["out_dim"]):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = torch.randn(in_dim, out_dim)

    def forward(self, x: Float[Tensor, "b in_dim"]) -> Float[Tensor, "b out_dim"]:
        return x @ self.weight


def test_dense_layers(x: Float[Tensor, "batch 128"], y: Float[Tensor, "batch 64"]):
    dense1 = MyDense(128, 256)
    dense2 = MyDense(64, 128)

    out1 = dense1.forward(x)
    out2 = dense2.forward(y)

    return out1, out2
