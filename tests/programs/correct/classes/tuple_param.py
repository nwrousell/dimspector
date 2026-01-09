from jaxtyping import Float
import torch
from torch import Tensor


class MultiHeadAttention:
    def __init__(
        self,
        dims: tuple[int["d_model"], int["d_k"], int["d_v"]],
        num_heads: int["n_heads"],
    ):
        self.d_model, self.d_k, self.d_v = dims
        self.num_heads = num_heads
        self.q_weight = torch.randn(self.d_model, self.num_heads * self.d_k)
        self.k_weight = torch.randn(self.d_model, self.num_heads * self.d_k)
        self.v_weight = torch.randn(self.d_model, self.num_heads * self.d_v)

    def forward(
        self, x: Float[Tensor, "b seq_len d_model"]
    ) -> Float[Tensor, "b seq_len d_model"]:
        q = x @ self.q_weight
        k = x @ self.k_weight
        v = x @ self.v_weight
        return q + k + v  # Simplified


def test_attention(x: Float[Tensor, "batch 10 512"]):
    attn = MultiHeadAttention((512, 64, 64), 8)
    y = attn.forward(x)
