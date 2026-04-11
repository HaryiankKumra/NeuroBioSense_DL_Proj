"""Physiological signal encoder for multimodal emotion recognition.

This module processes 6-channel Empatica E4 signals:
BVP, EDA, TEMP, ACC_X, ACC_Y, ACC_Z.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention_module import TemporalAttentionPool


class ChannelAttention(nn.Module):
    """Lightweight channel-wise attention over physiological modalities."""

    def __init__(self, channels: int = 6) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel gates.

        Args:
            x: Tensor of shape (B, T_s, 6).

        Returns:
            Tensor of shape (B, T_s, 6).
        """
        pooled = x.mean(dim=1)  # (B, T_s, 6) -> (B, 6)
        gates = self.fc1(pooled)  # (B, 6) -> (B, 6)
        gates = self.relu(gates)  # (B, 6) -> (B, 6)
        gates = self.fc2(gates)  # (B, 6) -> (B, 6)
        gates = self.sigmoid(gates)  # (B, 6) -> (B, 6)
        out = x * gates.unsqueeze(1)  # (B, T_s, 6) * (B, 1, 6) -> (B, T_s, 6)
        return out


class SignalCNNBlocks(nn.Module):
    """Two-block Conv1D encoder over channel-attended signals."""

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal windows.

        Args:
            x: Tensor of shape (B, T_s, 6).

        Returns:
            Tensor of shape (B, T_s//4, 64).
        """
        conv_in = x.transpose(1, 2)  # (B, T_s, 6) -> (B, 6, T_s)
        block1_out = self.block1(conv_in)  # (B, 6, T_s) -> (B, 32, T_s//2)
        block2_out = self.block2(block1_out)  # (B, 32, T_s//2) -> (B, 64, T_s//4)
        out = block2_out.transpose(1, 2)  # (B, 64, T_s//4) -> (B, T_s//4, 64)
        return out


class SignalModule(nn.Module):
    """Full signal pipeline: ChannelAttention + CNN + BiLSTM + AttentionPool."""

    def __init__(self, channels: int = 6) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels=channels)
        self.cnn_blocks = SignalCNNBlocks(in_channels=channels)

        # WHY bidirectional + 2 layers: physiology trends evolve over time,
        # and deeper recurrent capacity helps disambiguate subtle affect cues.
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.temporal_attention = TemporalAttentionPool(input_dim=256)

    def forward(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode physiological windows.

        Args:
            signal: Tensor with shape (B, T_s, 6).

        Returns:
            sig_emb: Tensor with shape (B, 256).
            lstm_out: Tensor with shape (B, T_s//4, 256).
            attn_weights: Tensor with shape (B, T_s//4).
        """
        attended = self.channel_attention(signal)  # (B, T_s, 6) -> (B, T_s, 6)
        cnn_out = self.cnn_blocks(attended)  # (B, T_s, 6) -> (B, T_s//4, 64)
        lstm_out, _ = self.bilstm(cnn_out)  # (B, T_s//4, 64) -> (B, T_s//4, 256)
        sig_emb, attn_weights = self.temporal_attention(lstm_out)  # (B, T_s//4, 256) -> (B, 256), (B, T_s//4)
        return sig_emb, lstm_out, attn_weights

    def freeze_cnn_blocks(self) -> None:
        """Freeze only convolutional blocks for Stage 3 fine-tuning."""
        for param in self.cnn_blocks.parameters():
            param.requires_grad = False

    def set_stage3_policy(self) -> None:
        """Stage 3 policy: frozen CNN, trainable recurrent-attention head."""
        self.freeze_cnn_blocks()
        for param in self.channel_attention.parameters():
            param.requires_grad = True
        for param in self.bilstm.parameters():
            param.requires_grad = True
        for param in self.temporal_attention.parameters():
            param.requires_grad = True
