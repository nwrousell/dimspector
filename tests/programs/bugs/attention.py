from jaxtyping import Float
import torch
from torch import Tensor


def attention_wrong_seq(
    scores: Float[Tensor, "batch heads seq_q seq_k"], v: Float[Tensor, "batch heads seq_v d"]
):
    out = scores @ v
    return out
