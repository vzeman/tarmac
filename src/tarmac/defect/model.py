from __future__ import annotations

import torch
import torch.nn as nn


class DefectHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)

