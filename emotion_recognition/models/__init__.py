"""Model package exports for the NeuroBioSense multimodal pipeline."""

from .attention_module import CrossModalAttention, TemporalAttentionPool
from .classifier import EmotionClassifier
from .face_module import FaceModule
from .facenet_backbone import FaceNetBackbone
from .full_model import MultimodalEmotionModel
from .fusion_module import SoftGatingFusion
from .projection_head import ProjectionHead
from .signal_module import ChannelAttention, SignalCNNBlocks, SignalModule

__all__ = [
    "FaceNetBackbone",
    "ProjectionHead",
    "FaceModule",
    "ChannelAttention",
    "SignalCNNBlocks",
    "SignalModule",
    "TemporalAttentionPool",
    "CrossModalAttention",
    "SoftGatingFusion",
    "EmotionClassifier",
    "MultimodalEmotionModel",
]
