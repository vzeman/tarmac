from __future__ import annotations

import torch
import torch.nn as nn


class CrackHead(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings).squeeze(1)
