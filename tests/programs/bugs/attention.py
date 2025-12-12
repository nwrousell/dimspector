from typing import Generic
import torch


class T(Generic(str)): ...


def attention_wrong_seq(
    scores: T["batch heads seq_q seq_k"], v: T["batch heads seq_v d"]
):
    out = scores @ v
    return out
