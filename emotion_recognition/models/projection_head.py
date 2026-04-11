"""Projection head for adapting FaceNet embeddings to emotion-specific features.

The projection head is intentionally lightweight and always trainable to bridge
identity-centric face embeddings to emotion-centric representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """512 -> 256 -> 128 projection with regularization."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            # WHY dropout=0.3: strong regularization for small-scale affective data.
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings.

        Args:
            x: Tensor of shape (N, 512).

        Returns:
            Tensor of shape (N, 128).
        """
        out = self.net(x)  # (N, 512) -> (N, 128)
        return out
