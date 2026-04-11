"""Attention modules used by face, signal, and fusion pipelines.

Includes:
- TemporalAttentionPool for sequence compression
- CrossModalAttention for bidirectional video-signal refinement
"""

from __future__ import annotations

from math import sqrt
from typing import Tuple

import torch
import torch.nn as nn


class TemporalAttentionPool(nn.Module):
    """Temporal attention pooling over sequence features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.score_layer = nn.Linear(input_dim, 1)

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool sequence features with learned temporal salience.

        Args:
            seq: Tensor of shape (B, T, D).

        Returns:
            pooled: Tensor of shape (B, D).
            weights: Tensor of shape (B, T).
        """
        scores = self.score_layer(seq).squeeze(-1)  # (B, T, D) -> (B, T, 1) -> (B, T)
        weights = torch.softmax(scores, dim=1)  # (B, T) -> (B, T)
        pooled = (weights.unsqueeze(-1) * seq).sum(dim=1)  # (B, T, 1)*(B, T, D) -> (B, D)
        return pooled, weights


class CrossModalAttention(nn.Module):
    """Bidirectional cross-modal attention between video and signal embeddings."""

    def __init__(self, vid_dim: int = 128, sig_dim: int = 256, attn_dim: int = 128) -> None:
        super().__init__()
        self.scale = sqrt(attn_dim)

        self.vid_q = nn.Linear(vid_dim, attn_dim)
        self.vid_k = nn.Linear(vid_dim, attn_dim)
        self.vid_v = nn.Linear(vid_dim, attn_dim)

        self.sig_q = nn.Linear(sig_dim, attn_dim)
        self.sig_k = nn.Linear(sig_dim, attn_dim)
        self.sig_v = nn.Linear(sig_dim, attn_dim)

        # WHY projection: signal embedding is 256-d, but cross-attention operates
        # in a shared 128-d latent space for symmetric interactions.
        self.sig_residual_proj = nn.Linear(sig_dim, attn_dim)
        self.sig_back_proj = nn.Linear(attn_dim, sig_dim)

    def forward(self, vid_emb: torch.Tensor, sig_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run bidirectional cross-modal attention.

        Args:
            vid_emb: Tensor of shape (B, 128).
            sig_emb: Tensor of shape (B, 256).

        Returns:
            enhanced_vid: Tensor of shape (B, 128).
            enhanced_sig: Tensor of shape (B, 256).
        """
        vid_q = self.vid_q(vid_emb).unsqueeze(1)  # (B, 128) -> (B, 1, 128)
        sig_k = self.sig_k(sig_emb).unsqueeze(1)  # (B, 256) -> (B, 1, 128)
        sig_v = self.sig_v(sig_emb).unsqueeze(1)  # (B, 256) -> (B, 1, 128)

        vid_k = self.vid_k(vid_emb).unsqueeze(1)  # (B, 128) -> (B, 1, 128)
        vid_v = self.vid_v(vid_emb).unsqueeze(1)  # (B, 128) -> (B, 1, 128)
        sig_q = self.sig_q(sig_emb).unsqueeze(1)  # (B, 256) -> (B, 1, 128)

        # WHY video->signal attention: visual branch can query physiological context
        # to reduce ambiguity when facial expressions are subtle.
        logits_vid = torch.matmul(vid_q, sig_k.transpose(1, 2)) / self.scale  # (B, 1, 128)x(B, 128, 1) -> (B, 1, 1)
        weights_vid = torch.softmax(logits_vid, dim=-1)  # (B, 1, 1) -> (B, 1, 1)
        attn_vid = torch.matmul(weights_vid, sig_v)  # (B, 1, 1)x(B, 1, 128) -> (B, 1, 128)
        enhanced_vid = vid_emb + attn_vid.squeeze(1)  # (B, 128) + (B, 128) -> (B, 128)

        # WHY signal->video attention: physiology can focus on expression-relevant
        # visual content, improving robustness to sensor noise.
        logits_sig = torch.matmul(sig_q, vid_k.transpose(1, 2)) / self.scale  # (B, 1, 128)x(B, 128, 1) -> (B, 1, 1)
        weights_sig = torch.softmax(logits_sig, dim=-1)  # (B, 1, 1) -> (B, 1, 1)
        attn_sig = torch.matmul(weights_sig, vid_v)  # (B, 1, 1)x(B, 1, 128) -> (B, 1, 128)

        sig_base = self.sig_residual_proj(sig_emb)  # (B, 256) -> (B, 128)
        enhanced_sig_128 = sig_base + attn_sig.squeeze(1)  # (B, 128) + (B, 128) -> (B, 128)
        enhanced_sig = self.sig_back_proj(enhanced_sig_128)  # (B, 128) -> (B, 256)

        return enhanced_vid, enhanced_sig
