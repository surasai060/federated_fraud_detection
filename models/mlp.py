"""
models/mlp.py
-------------
The local model used by every federated client.
A simple Multi-Layer Perceptron (MLP) for binary fraud classification.

Architecture:
  Input (30 features) → 64 → 32 → 16 → 1 (sigmoid output)
"""

import torch
import torch.nn as nn


class FraudMLP(nn.Module):
    """
    Binary classification MLP.
    Output is a single logit (use BCEWithLogitsLoss during training).
    """

    def __init__(self, input_dim: int = 30, dropout: float = 0.3):
        super(FraudMLP, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)   # Raw logit — no sigmoid here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def get_model(input_dim: int = 30) -> FraudMLP:
    """Factory function — returns a fresh model instance."""
    return FraudMLP(input_dim=input_dim)


def get_parameters(model: nn.Module):
    """Extract model weights as a list of numpy arrays (for Flower)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
    """Load weights from a list of numpy arrays into model (for Flower)."""
    import numpy as np
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
