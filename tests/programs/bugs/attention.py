from typing import Generic
import torch


class T(Generic(str)): ...


# def attention_kv_mismatch(
#     scores: T["batch heads seq seq"], v: T["batch heads seq d_v"]
# ):
#     """Error: scores @ v requires scores' last dim to match v's second-to-last dim.
#     scores is [batch, heads, seq, seq] and v is [batch, heads, seq, d_v].
#     This actually works! But let's make it wrong:
#     """
#     pass


def attention_wrong_seq(
    scores: T["batch heads seq_q seq_k"], v: T["batch heads seq_v d"]
):
    out = scores @ v
    return out
