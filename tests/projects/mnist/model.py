from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor
from layers import init_weight_matrix, init_bias_vector


class LinearLayer(nn.Module):
    def __init__(self, in_features: int["in"], out_features: int["out"]):
        super().__init__()
        self.weight = init_weight_matrix(in_features, out_features)
        self.bias = init_bias_vector(out_features)

    def forward(self, x: Float[Tensor, "batch in"]) -> Float[Tensor, "batch out"]:
        """Apply a linear transformation: x @ weight.T + bias."""
        return x @ torch.transpose(self.weight) + self.bias


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int["784"],
        hidden_size: int["hidden"],
        num_classes: int["classes"],
    ):
        super().__init__()
        self.layer1 = LinearLayer(input_size, hidden_size)
        self.layer2 = LinearLayer(hidden_size, hidden_size)
        self.layer3 = LinearLayer(hidden_size, num_classes)

    def forward(
        self, x: Float[Tensor, "batch 1 28 28"]
    ) -> Float[Tensor, "batch classes"]:
        """Forward pass through the MLP."""
        x = torch.view(x, (x.shape[0], -1))
        x = torch.nn.functional.relu(self.layer1.forward(x))
        x = torch.nn.functional.relu(self.layer2.forward(x))
        x = self.layer3.forward(x)
        return x
