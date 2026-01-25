from jaxtyping import Float
import torch
from torch import Tensor
from model import LinearModel
from typing import Tuple

def train_model(
    in_features: int, out_features: int, batch_size: int
) -> Float[Tensor, "32"]:
    """Initialize and train a model."""
    # Initialize model
    model = LinearModel(in_features, out_features)

    # Create dummy data
    x = torch.randn(batch_size, in_features)
    y = torch.randn(batch_size, out_features)

    # Forward pass
    output = model(x)

    # Compute loss
    loss = torch.nn.functional.mse_loss(output, y)
    return loss


# def evaluate_model(
#     model: LinearModel, batch_size: int, in_features: int
# ) -> Float[Tensor, "batch out"]:
#     """Evaluate a model."""
#     x = torch.randn(batch_size, in_features)
#     output = model.forward(x)
#     return output
