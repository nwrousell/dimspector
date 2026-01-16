from typing import Generic
import torch


class T(Generic(str)): ...


def relative_attention_cross(
    x_q: T["batch seq_q d_model"],
    x_kv: T["batch seq_k d_model"],
    W_q: T["d_model heads d_k"],
    W_k: T["d_model heads d_k"],
    W_v: T["d_model heads d_v"],
    rel_pos_bias: T["seq_k seq_q"],  # Bug: dimensions transposed
):
    """Cross-attention with relative position bias."""
    # Project inputs to query, key, value
    query = torch.transpose(
        torch.reshape(
            x_q @ torch.reshape(W_q, (x_q.shape[2], -1)),
            (x_q.shape[0], x_q.shape[1], W_q.shape[1], W_q.shape[2]),
        ),
        1,
        2,
    )
    key = torch.transpose(
        torch.reshape(
            x_kv @ torch.reshape(W_k, (x_kv.shape[2], -1)),
            (x_kv.shape[0], x_kv.shape[1], W_k.shape[1], W_k.shape[2]),
        ),
        1,
        2,
    )
    value = torch.transpose(
        torch.reshape(
            x_kv @ torch.reshape(W_v, (x_kv.shape[2], -1)),
            (x_kv.shape[0], x_kv.shape[1], W_v.shape[1], W_v.shape[2]),
        ),
        1,
        2,
    )

    # Compute attention scores
    scores = query @ torch.transpose(key, -1, -2)

    # Add relative position bias
    biased = scores + rel_pos_bias

    weights = torch.softmax(biased, dim=-1)
    out = weights @ value
    return out
