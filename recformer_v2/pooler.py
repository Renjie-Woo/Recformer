import torch
import torch.nn as nn


class RecformerPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_status: torch.Tensor) -> torch.Tensor:
        return hidden_status[:, 0]
