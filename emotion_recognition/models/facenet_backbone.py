"""FaceNet backbone wrapper for frame-level facial representation learning.

This module centralizes loading and freezing logic for InceptionResnetV1
(from facenet-pytorch) so training scripts can enforce stage-specific
policies without duplicating parameter-selection code.
"""

from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError as exc:
    raise ImportError(
        "facenet-pytorch is required. Install with: pip install facenet-pytorch"
    ) from exc


class FaceNetBackbone(nn.Module):
    """InceptionResnetV1 backbone with explicit freeze-policy controls."""

    # WHY these keywords: the last high-level residual stack and terminal layers
    # are the most task-specific, so Stage 1 fine-tuning only updates these.
    DEFAULT_UNFREEZE_KEYWORDS: List[str] = [
        "repeat_3",
        "block8",
        "last_linear",
        "last_bn",
    ]

    def __init__(self, pretrained: str = "vggface2") -> None:
        super().__init__()
        # WHY classify=False: we need 512-d embeddings, not identity logits.
        self.model = InceptionResnetV1(pretrained=pretrained, classify=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute 512-d embeddings.

        Args:
            x: Face batch tensor, shape (N, 3, 160, 160).

        Returns:
            Tensor of shape (N, 512).
        """
        emb = self.model(x)  # (N, 3, 160, 160) -> (N, 512)
        return emb

    def freeze_all(self) -> None:
        """Freeze every backbone parameter."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_by_keywords(self, keywords: Iterable[str]) -> None:
        """Unfreeze parameters whose names contain any keyword."""
        keywords = list(keywords)
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in keywords):
                param.requires_grad = True

    def unfreeze_last_inception_block(self) -> None:
        """Unfreeze only the terminal representation blocks for Stage 1."""
        self.unfreeze_by_keywords(self.DEFAULT_UNFREEZE_KEYWORDS)

    def set_stage1_policy(self) -> None:
        """Stage 1 policy: freeze all, then unfreeze the last inception block."""
        self.freeze_all()
        self.unfreeze_last_inception_block()

    def set_stage3_policy(self) -> None:
        """Stage 3 policy: fully frozen backbone for stable transfer."""
        self.freeze_all()

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only parameters marked trainable for optimizer grouping."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_trainable_parameters(self) -> int:
        """Convenience count used for training logs."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
