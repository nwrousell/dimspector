from jaxtyping import Float
import torch
from torch import Tensor


def relative_attention_cross(
    x_q: Float[Tensor, "batch seq_q d_model"],
    x_kv: Float[Tensor, "batch seq_k d_model"],
    W_q: Float[Tensor, "d_model heads d_k"],
    W_k: Float[Tensor, "d_model heads d_k"],
    W_v: Float[Tensor, "d_model heads d_v"],
    rel_pos_bias: Float[Tensor, "seq_k seq_q"],
):
    # Extract dimensions
    batch = x_q.shape[0]
    seq_q = x_q.shape[1]
    seq_k = x_kv.shape[1]
    d_model = x_q.shape[2]
    heads = W_q.shape[1]
    d_k = W_q.shape[2]
    d_v = W_v.shape[2]

    # Project inputs to query, key, value
    query = torch.transpose(
        torch.reshape(x_q @ torch.reshape(W_q, (d_model, -1)), (batch, seq_q, heads, d_k)),
        1, 2
    )
    key = torch.transpose(
        torch.reshape(x_kv @ torch.reshape(W_k, (d_model, -1)), (batch, seq_k, heads, d_k)),
        1 ,2
    )
    value = torch.transpose(
        torch.reshape(x_kv @ torch.reshape(W_v, (d_model, -1)), (batch, seq_k, heads, d_v)),
        1, 2,
    )

    # Compute attention scores
    scores = query @ torch.transpose(key, -1, -2)

    # Add relative position bias
    biased = scores + rel_pos_bias

    weights = torch.softmax(biased, dim=-1)
    out = weights @ value
    return out
