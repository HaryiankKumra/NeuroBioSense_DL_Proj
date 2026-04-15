"""End-to-end multimodal emotion recognition model.

This module assembles:
- Face pipeline (FaceNet + projection + temporal BiLSTM + temporal attention)
- Signal pipeline (channel attention + CNN + BiLSTM + temporal attention)
- Cross-modal attention
- Soft-gating fusion
- Final classifier
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .attention_module import CrossModalAttention
from .classifier import EmotionClassifier
from .face_module import FaceModule
from .fusion_module import SoftGatingFusion
from .signal_module import SignalModule


class MultimodalEmotionModel(nn.Module):
    """Research-level multimodal architecture for NeuroBioSense."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.face_module = FaceModule()
        self.signal_module = SignalModule(channels=6)
        self.cross_modal_attention = CrossModalAttention(vid_dim=128, sig_dim=256, attn_dim=128)
        self.fusion = SoftGatingFusion(vid_dim=128, sig_dim=256, fused_dim=384)
        self.classifier = EmotionClassifier(input_dim=384, num_classes=self.num_classes)

    def apply_stage3_freezing(self) -> None:
        """Apply the Stage 3 trainable/frozen policy."""
        self.face_module.set_stage3_policy()
        self.signal_module.set_stage3_policy()

    def forward(
        self,
        video: torch.Tensor,
        signal: torch.Tensor,
        use_face: bool = True,
        use_signal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            video: Tensor of shape (B, T_v, 3, 160, 160).
            signal: Tensor of shape (B, T_s, 6).
            use_face: Whether to execute face branch (False uses zero face embeddings).
            use_signal: Whether to execute signal branch (False uses zero signal embeddings).

        Returns:
            output: Log-probabilities tensor of shape (B, C).
            confidence: Scalar for B=1 else tensor shape (B,).
        """
        bsz, t_v = video.shape[0], video.shape[1]

        if use_face:
            facenet_emb, vid_emb, _ = self.face_module(video)  # (B, T_v, 3, 160, 160) -> (B, T_v, 128), (B, 128)
            assert facenet_emb.shape == (bsz, t_v, 128), f"Expected {(bsz, t_v, 128)}, got {tuple(facenet_emb.shape)}"
        else:
            facenet_emb = torch.zeros((bsz, t_v, 128), device=video.device, dtype=video.dtype)
            vid_emb = torch.zeros((bsz, 128), device=video.device, dtype=video.dtype)

        if use_signal:
            sig_emb, _, _ = self.signal_module(signal)  # (B, T_s, 6) -> (B, 256)
        else:
            sig_emb = torch.zeros((bsz, 256), device=video.device, dtype=video.dtype)

        assert vid_emb.shape == (bsz, 128), f"Expected {(bsz, 128)}, got {tuple(vid_emb.shape)}"
        assert sig_emb.shape == (bsz, 256), f"Expected {(bsz, 256)}, got {tuple(sig_emb.shape)}"

        enhanced_vid, enhanced_sig = self.cross_modal_attention(vid_emb, sig_emb)  # (B, 128), (B, 256) -> (B, 128), (B, 256)
        fused, _ = self.fusion(enhanced_vid, enhanced_sig)  # (B, 128)+(B, 256) -> (B, 384)
        assert fused.shape == (bsz, 384), f"Expected {(bsz, 384)}, got {tuple(fused.shape)}"

        output = self.classifier(fused)  # (B, 384) -> (B, C)
        assert output.shape == (bsz, self.num_classes), f"Expected {(bsz, self.num_classes)}, got {tuple(output.shape)}"

        probs = torch.exp(output)  # (B, C) -> (B, C)
        confidence_vec = probs.max(dim=1).values  # (B, C) -> (B,)
        confidence = confidence_vec.squeeze(0) if confidence_vec.numel() == 1 else confidence_vec

        return output, confidence


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MultimodalEmotionModel().to(device)

    B, T_v, T_s, C = 1, 10, 128, 6
    video = torch.randn(B, T_v, 3, 160, 160).to(device)
    signal = torch.randn(B, T_s, C).to(device)

    with torch.inference_mode():
        output, confidence = model(video, signal)

    print(f"Output shape : {output.shape}")
    print(f"Confidence   : {float(confidence):.4f}")
    print(f"Predicted    : {output.argmax(dim=1).item()}")
    assert output.shape == (1, 7), "Shape mismatch!"
    print("All assertions passed.")
