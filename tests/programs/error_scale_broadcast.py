from typing import Generic
import torch


class T(Generic(str)): ...


def scale_wrong_channels(x: T["batch channels h w"], scale: T["other_channels"]):
    """Error: scale has 'other_channels' but x has 'channels' - not broadcastable."""
    out = x * scale
    return out
